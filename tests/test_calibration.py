import os
from unittest import mock
from tempfile import TemporaryDirectory


from utils import get_mock_model, in_memory_db_factory

get_in_memory_db = in_memory_db_factory()


@mock.patch("autumn.db.database._get_sql_engine", get_in_memory_db)
def test_calibrate_autumn_mcmc():
    # Import autumn stuff inside function so we can mock out the database.
    from autumn.db import Database
    from autumn.calibration import Calibration, CalibrationMode

    priors = [{"param_name": "ice_cream_sales", "distribution": "uniform", "distri_params": [1, 5]}]
    target_outputs = [
        {
            "output_key": "shark_attacks",
            "years": [2000, 2001, 2002, 2003, 2004],
            "values": [3, 6, 9, 12, 15],
            "loglikelihood_distri": "poisson",
        }
    ]

    def build_mock_model(params):
        """
        Fake model building function where derived output "shark_attacks" 
        is influenced by the ice_cream_sales input parameter.
        """
        ice_cream_sales = params["ice_cream_sales"]
        vals = [0, 1, 2, 3, 4, 5]
        mock_model = get_mock_model(
            times=[1999, 2000, 2001, 2002, 2003, 2004],
            outputs=[
                [300.0, 300.0, 300.0, 33.0, 33.0, 33.0, 93.0, 39.0],
                [271.0, 300.0, 271.0, 62.0, 33.0, 62.0, 93.0, 69.0],
                [246.0, 300.0, 246.0, 88.0, 33.0, 88.0, 93.0, 89.0],
                [222.0, 300.0, 222.0, 111.0, 33.0, 111.0, 39.0, 119.0],
                [201.0, 300.0, 201.0, 132.0, 33.0, 132.0, 39.0, 139.0],
                [182.0, 300.0, 182.0, 151.0, 33.0, 151.0, 39.0, 159.0],
            ],
            derived_outputs={
                "times": [1999, 2000, 2001, 2002, 2003, 2004],
                "shark_attacks": [ice_cream_sales * i for i in vals],
            },
        )
        return mock_model

    with TemporaryDirectory() as dirpath:
        with mock.patch("autumn.calibration.calibration.constants.DATA_PATH", dirpath):
            calib = Calibration(
                "sharks",
                build_mock_model,
                priors,
                target_outputs,
                multipliers={},
                chain_index=0,
                model_parameters={"default": {}, "scenario_start_time": 2000, "scenarios": {}},
            )
            num_iters = 100
            calib.run_fitting_algorithm(
                run_mode=CalibrationMode.AUTUMN_MCMC,
                n_iterations=num_iters,
                n_burned=20,
                n_chains=1,
                available_time=1e6,
            )
            app_dir = os.path.join(dirpath, "sharks")
            run_dir = os.path.join(app_dir, os.listdir(app_dir)[0])
            out_db_path = os.path.join(run_dir, os.listdir(run_dir)[0])
            assert os.path.exists(out_db_path)

            out_db = Database(out_db_path)
            mcmc_runs = out_db.db_query("mcmc_run")
            max_idx = mcmc_runs.loglikelihood.idxmax()
            best_run = mcmc_runs.iloc[max_idx]
            ice_cream_sales_mle = best_run.ice_cream_sales
            # assert 2.6 < ice_cream_sales_mle < 3.4
