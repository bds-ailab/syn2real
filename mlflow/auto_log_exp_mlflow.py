import yaml
import mlflow
import os


def search_unlogged_exp_to_be_logged():
    """search for unlogged experiments in /out/ volume that has the flag 'to_log=True'

    Returns:
        unlogged_experiments (list): list of unlogged experiments yaml files
        exp_paths (list): list of yaml files paths
    """
    # Initialize empty lists
    unlogged_experiments = []
    exp_paths = []

    # Read /out/ volume folders
    folders = os.listdir("/out/")
    for f in folders:
        # Look for experiments folders that always have these two yml files
        if (
            os.path.isdir("/out/" + f)
            and "exp.yml" in os.listdir("/out/" + f)
            and "config.yml" in os.listdir("/out/" + f)
        ):
            # Read experiment resume file
            with open(f"/out/{f}/exp.yml", "r") as exp_file:
                exp = yaml.safe_load(exp_file)

            # Ensure that the experiment is wanted to be logged and is not yet logged
            if exp["to_log"] and not (exp["logged"]):
                unlogged_experiments.append(exp)
                exp_paths.append(f"/out/{f}/exp.yml")
    return unlogged_experiments, exp_paths


def log_exp(exp_ymls, exp_paths, mlflow_uri="https://mlflow.sf.eviden.com/"):
    """log found experiments with their configuration parameters and weights files

    Args:
        exp_ymls (list): list of unlogged experiments yaml files
        exp_paths (list): list of yaml files paths
        mlflow_uri (str, optional): mlflow tracking uri. Defaults to "https://mlflow.sf.eviden.com/".
    """

    # Set mlflow tracking uri
    mlflow.set_tracking_uri(mlflow_uri)

    # Iterate on each found experiment
    for i in range(len(exp_ymls)):

        # Set experiment name from resume file
        mlflow.set_experiment(exp_ymls[i]["exp_name"])

        # Set run name from resume file
        with mlflow.start_run(run_name=exp_ymls[i]["run_name"]):

            # Log weights file
            print("logging model weights")
            path = f'/models/{exp_ymls[i]["exp_name"]}_{exp_ymls[i]["run_name"]}/'
            for part in os.listdir(path):
                mlflow.log_artifact(path + part)

            # Log params
            print("logging training configuration")
            with open(exp_ymls[i]["config_file"], "r") as yaml_file:
                yaml_content = yaml.safe_load(yaml_file)
                for k, v in yaml_content.items():
                    mlflow.log_param(k, v)

            # Set flag to logged
            exp_ymls[i]["logged"] = True

            # Update the experiment resume file to save the "logged" flag change
            with open(exp_paths[i], "w") as yaml_file:
                yaml.dump(exp_ymls[i], yaml_file, default_flow_style=False)

        print(
            f"{exp_ymls[i]['exp_name']}_{exp_ymls[i]['run_name']} is logged successfully"
        )


if __name__ == "__main__":
    # Look for unlogged exp
    exp_ymls, exp_paths = search_unlogged_exp_to_be_logged()

    # Display search results
    if exp_ymls:
        print("Found following unlogged experiments : ")
        for exp in exp_ymls:
            print(f"{exp['exp_name']}_{exp['run_name']}")
    else:
        print("All experiments ready to be logged are logged :)")

    # Log found exp
    log_exp(exp_ymls, exp_paths)
