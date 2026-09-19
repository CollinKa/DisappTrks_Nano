"""LPC-flavored PocketCoffea manual HTCondor executor.

This module is meant to be passed to PocketCoffea with
``--executor-custom-setup lpc_condor_executor.py``.  It keeps PocketCoffea's
manual-job machinery -- fileset splitting, per-job pickled Configurators,
``job_*.idle/running/done/failed`` flags, and mergeable ``output_job_*.coffea``
files -- but writes a plain LPC Condor submit description instead of the
LXPLUS-specific one that requests a Singularity image.
"""

from __future__ import annotations

import os
import shlex

import pocket_coffea

from pocket_coffea.executors.executors_base import (
    FuturesExecutorFactory,
    IterativeExecutorFactory,
)
from pocket_coffea.executors.executors_lxplus import ExecutorFactoryCondorCERN
from pocket_coffea.executors.executors_manual_jobs import write_inner_run_options


class ExecutorFactoryCondorLPC(ExecutorFactoryCondorCERN):
    """Manual-job Condor executor with LPC-native submit files."""

    def __init__(self, run_options, outputdir, **kwargs):
        run_options.setdefault("cores-per-worker", 1)
        run_options.setdefault("mem-per-worker", "4GB")
        run_options.setdefault("disk-per-worker", "2GB")
        run_options.setdefault("queue", "workday")
        run_options.setdefault("local-virtualenv", True)
        run_options.setdefault("conda-env", False)
        run_options.setdefault("split-by-category", False)
        run_options.setdefault("eos-prefix", "root://cmseos.fnal.gov/")
        super().__init__(run_options, outputdir, **kwargs)

    def submit_jobs(self, jobs_config):
        """Prepare LPC Condor jobs and submit them.

        This is intentionally close to PocketCoffea's ``condor@lxplus``
        implementation, but omits ``MY.SingularityImage`` and uses ``python3``
        in the wrapper.  The worker environment comes from
        ``custom-setup-commands`` plus ``local-virtualenv: true`` in the run
        options, so the job runs in the same CMSSW/venv style as legacy LPC
        DisappTrks jobs.
        """

        abs_output_path = os.path.abspath(self.outputdir)
        abs_jobdir_path = os.path.abspath(self.jobs_dir)
        os.makedirs(f"{self.jobs_dir}/logs", exist_ok=True)

        env_extras = "\n".join(self._worker_env())

        if os.getenv("PYTHONPATH"):
            pythonpath = os.getenv("PYTHONPATH")
        else:
            pythonpath = "/".join(pocket_coffea.__file__.split("/")[:-2])

        copy_command = "cp"
        eos_prefix = self.run_options.get("eos-prefix", "root://cmseos.fnal.gov/")
        if abs_output_path.startswith("/eos/"):
            abs_output_path = eos_prefix + abs_output_path
        if abs_output_path.startswith(eos_prefix):
            copy_command = "xrdcp -f"

        runnerpath = f"{pythonpath}/pocket_coffea/scripts/runner.py"
        if os.path.isfile(runnerpath):
            runnercmd = f"python3 {runnerpath}"
        else:
            runnercmd = "pocket-coffea run"

        inner_yaml_path = write_inner_run_options(self.jobs_dir, self.run_options)
        inner_yaml_basename = os.path.basename(inner_yaml_path)

        if self.run_options["split-by-category"]:
            splitcommands = f"""
    cd {abs_output_path}
    split-output output_all.coffea -b category -o output.coffea
    rm output_all.coffea
    for f in *.coffea; do
        run_with_retries "$1" "{copy_command} $f {abs_output_path}/${{f%.coffea}}_job_$1.coffea"
    done
"""
        else:
            splitcommands = (
                f'run_with_retries "$1" "{copy_command} output/output_all.coffea '
                f'{abs_output_path}/output_job_$1.coffea"'
            )

        script = f"""#!/bin/bash
set -e
{env_extras}

JOBDIR={abs_jobdir_path}

run_with_retries() {{
    local jobid="$1"
    shift
    local cmd="$*"
    for i in {{1..10}}; do
        eval "$cmd" && return 0
        sleep 10
    done
    echo "$cmd failed after 10 attempts."
    rm -f "$JOBDIR/job_$jobid.running"
    touch "$JOBDIR/job_$jobid.failed"
    exit 1
}}

rm -f "$JOBDIR/job_$1.idle"

echo "Starting job $1 on $(hostname)"
touch "$JOBDIR/job_$1.running"

{runnercmd} --cfg "$2" -o output EXECUTOR --chunksize "$3" --custom-run-options {inner_yaml_basename}
status=$?

if [ "$status" -eq 0 ]; then
    echo 'Job successful'
    {splitcommands}
    rm -f "$JOBDIR/job_$1.running"
    touch "$JOBDIR/job_$1.done"
else
    echo 'Job failed'
    rm -f "$JOBDIR/job_$1.running"
    touch "$JOBDIR/job_$1.failed"
    exit "$status"
fi
echo 'Done'
"""

        if int(self.run_options["cores-per-worker"]) > 1:
            script = script.replace(
                "EXECUTOR",
                f"--executor futures --scaleout {self.run_options['cores-per-worker']}",
            )
        else:
            script = script.replace("EXECUTOR", "--executor iterative")

        with open(f"{self.jobs_dir}/job.sh", "w") as f:
            f.write(script)

        chunksize_cfg = self.run_options["chunksize"]
        self._validate_chunksize_keys(chunksize_cfg, self.filesets)
        per_job_chunksize = [
            self._resolve_chunksize_for_job(chunksize_cfg, split)
            for split in self._splits
        ]

        sub = {
            "universe": "vanilla",
            "Executable": "job.sh",
            "Error": f"{abs_jobdir_path}/logs/job_$(ClusterId).$(ProcId).err",
            "Output": f"{abs_jobdir_path}/logs/job_$(ClusterId).$(ProcId).out",
            "Log": f"{abs_jobdir_path}/logs/job_$(ClusterId).log",
            "RequestCpus": self.run_options["cores-per-worker"],
            "RequestMemory": f"{self.run_options['mem-per-worker']}",
            "RequestDisk": f"{self.run_options['disk-per-worker']}",
            "+JobFlavour": f'"{self.run_options.get("queue", "workday")}"',
            "arguments": "$(ProcId) config_job_$(ProcId).pkl $(chunksize)",
            "should_transfer_files": "YES",
            "when_to_transfer_output": "ON_EXIT",
            "transfer_input_files": (
                f"{abs_jobdir_path}/config_job_$(ProcId).pkl,"
                f"{self.x509_path},"
                f"{abs_jobdir_path}/job.sh,"
                f"{abs_jobdir_path}/{inner_yaml_basename}"
            ),
            "on_exit_remove": "(ExitBySignal == False) && (ExitCode == 0)",
            "max_retries": self.run_options["retries"],
            "requirements": "Machine =!= LastRemoteHost",
        }

        with open(f"{self.jobs_dir}/jobs_all.sub", "w") as f:
            for key, value in sub.items():
                f.write(f"{key} = {value}\n")
            f.write("queue chunksize from (\n")
            for chunksize in per_job_chunksize:
                f.write(f"  {chunksize}\n")
            f.write(")\n")

        print(f"Creating {len(jobs_config)} .sub files for individual job submission.")
        for i, _ in enumerate(jobs_config):
            with open(f"{self.jobs_dir}/job_{i}.sub", "w") as f:
                for key, value in sub.items():
                    if isinstance(value, str):
                        value = value.replace("$(ProcId)", str(i))
                        value = value.replace("$(ClusterId).log", f"$(ClusterId).{i}.log")
                        value = value.replace("$(chunksize)", str(per_job_chunksize[i]))
                    f.write(f"{key} = {value}\n")
                f.write("queue\n")
            with open(f"{self.jobs_dir}/job_{i}.idle", "w") as f:
                f.write("")

        if self.run_options.get("dry-run", False):
            print(f"Dry run, not submitting jobs. You can find all files: {abs_jobdir_path}")
            return

        print("Submitting jobs")
        os.system(f"cd {abs_jobdir_path} && condor_submit jobs_all.sub")

    def _worker_env(self):
        env_worker = [
            "export XRD_RUNFORKHANDLER=1",
            "export MALLOC_TRIM_THRESHOLD_=0",
        ]
        if not self.run_options["ignore-grid-certificate"]:
            env_worker.append(f"export X509_USER_PROXY={self.x509_path}")
        for name in (
            "DISAPPTRKS_FIDUCIAL_MAP_DIR",
            "DISAPPTRKS_ELECTRON_FIDUCIAL_MAP_JSON",
            "DISAPPTRKS_MUON_FIDUCIAL_MAP_JSON",
            "DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS",
        ):
            if name in os.environ:
                env_worker.append(f"export {name}={shlex.quote(os.environ[name])}")
        if self.run_options.get("custom-setup-commands", None):
            env_worker += self.run_options["custom-setup-commands"]
        if self.run_options.get("local-virtualenv", False):
            env_worker.append(f"source {os.sys.prefix}/bin/activate")
            if os.getenv("PYTHONPATH"):
                pythonpath = os.getenv("PYTHONPATH")
            else:
                pythonpath = "/".join(pocket_coffea.__file__.split("/")[:-2])
            env_worker.append(f"export PYTHONPATH={pythonpath}:$PYTHONPATH")
        return env_worker


def get_executor_factory(executor_name, **kwargs):
    if executor_name == "iterative":
        return IterativeExecutorFactory(**kwargs)
    if executor_name == "futures":
        return FuturesExecutorFactory(**kwargs)
    if executor_name == "condor":
        return ExecutorFactoryCondorLPC(**kwargs)
    raise ValueError(f"Unsupported LPC executor: {executor_name}")
