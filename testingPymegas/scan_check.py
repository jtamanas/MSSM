import pandas as pd

import os
import sys
from tqdm.auto import tqdm

sys.path.append("/media/jt/data/Projects/MSSM/micromegas_5.2.7.a/MSSM/")
from pymicromegas import MicromegasSettings, SugraParameters
from softsusy import softsusy
from spheno import spheno
from suspect import suspect


def batch_simulator(
    df, settings, out_dir, num_total_sims=128, batch_size=32, clip_bound=23000
):
    """
    Run the Micromegas simulator on batches of sugra parameters.
    """

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        cnt = -1
        for i in range(0, len(lst), n):
            cnt += 1
            yield cnt, lst[i : i + n]

    def results_to_df(ins, res):
        out_cols = [
            "omega",
            "xf",
            "bsgsm",
            "bsgnlo",
            "deltarho",
            "bsmumu",
            "btaunu",
            "gmuon",
            "msne",
            "msnm",
            "msel",
            "mser",
            "msml",
            "msmr",
            "msdl",
            "msdr",
            "msul",
            "msur",
            "mssl",
            "mssr",
            "mscl",
            "mscr",
            "msnl",
            "msl1",
            "msl2",
            "msb1",
            "msb2",
            "mst1",
            "mst2",
            "mg",
            "mneut1",
            "mneut2",
            "mneut3",
            "mneut4",
            "mchg1",
            "mchg2",
            "mhsm",
            "mh",
            "mhc",
        ]
        res_dict = {}
        for key in ins.columns:
            res_dict[key] = ins[key].values

        for key in out_cols:
            values = getattr(res, key)()
            # the masses get recorded twice for some reason. this skips them
            print(values)
            repeated_val_increment = len(values) // len(res.omega())
            res_dict[key] = values[::repeated_val_increment]

        res_df = pd.DataFrame(res_dict)
        return res_df

    def save_results(results, out_dir, filename=None):
        """Save results to file."""
        if filename is None:
            filename = "results.csv"
        results.to_csv(os.path.join(out_dir, filename), index=False)

    for i, batch in tqdm(chunks(df[0:num_total_sims], batch_size)):
        # run the simulator
        # batch = batch.clip(-clip_bound, clip_bound)
        params = batch.apply(lambda x: SugraParameters(*x), axis=1).to_list()
        try:
            results = softsusy(params=params, settings=settings)
        except RuntimeError:
            # print(batch)
            print("micromegas failed")
            continue

        df_results = results_to_df(batch, results)
        save_results(df_results, out_dir, filename=f"{i}.csv")


if __name__ == "__main__":

    data_dir = "/media/jt/data/Projects/MSSM/data/Hollingsworth"
    sys.path.append(data_dir)
    from read_dataset import import_data

    # load and trim data
    dataset = "cMSSM"
    _df = import_data(os.path.join(data_dir, dataset), dataset.lower())
    _df = _df[_df["sgnmu"] == 1]
    sugras = _df[["m0", "m12", "a0", "tanb", "sgnmu"]]

    # default settings
    settings = MicromegasSettings(
        relic_density=True,
        masses=True,
        gmuon=True,
        bsg=True,
        bsmumu=True,
        btaunu=True,
        delta_rho=True,
        sort_odd=True,
        fast=True,
        beps=0.0001,
        cut=0.01,
    )

    out_dir = "/media/jt/data/Projects/MSSM/testingPymegas/scan_results/"

    # run the simulator
    batch_simulator(
        sugras,
        settings,
        out_dir=out_dir,
        num_total_sims=128*128,
        batch_size=128,
        clip_bound=20000,
    )
