from src.ReccurencePlot.RPlot import RecurrencePlotGenerator


"""def drop_sensors(data):
    tmp_data = data.loc[data["Normal/Attack"] == 1]
    for col in data.columns:
        print("col name: ", col, " unique: ", tmp_data[col].unique().shape[0])

    data.drop([data.columns[3], data.columns[4],
               data.columns[9], data.columns[10], data.columns[11], data.columns[12], data.columns[13],
               data.columns[14], data.columns[15],
               data.columns[22], data.columns[23],
               data.columns[28], data.columns[29], data.columns[30], data.columns[31], data.columns[32],
               data.columns[41], data.columns[42],
               data.columns[47], data.columns[48], data.columns[49]], axis=1, inplace=True)

    print(data.columns)

    data.to_csv(path + "Novikova_no_sensors.csv", sep=",", index=False)"""


if __name__ == "__main__":
    path = "C://Novikova/data/cutted/minus_5_param/"

    windows = [30]
    metric = "euclidean"
    mode = "binary"  # threshold binary
    threshold = 0.5

    for window in windows:
        print(f"Window: {window}")

        # Инициализация генератора
        rp_generator = RecurrencePlotGenerator(
            load_dir=path,
            save_dir=f"{path}{mode}_{metric}_{str(threshold).replace('.', '')}/Window_{window}/train",
            window=window,
            threshold=threshold,
            mode=mode,
            metric=metric,
            p=2.0,
            show_plot=False,
            save_plot=True,
            n_jobs=-1
        )

        rp_generator.load_and_prepare()
