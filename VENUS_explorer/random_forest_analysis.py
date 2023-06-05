"""
Analysis of the random forest on the dataset. 
    Date: May 2023
    Author: Wenhan Sun
"""

import numpy as np
import sklearn.ensemble
import VENUS_dataset
import matplotlib.pyplot as plt
import pandas as pd

def train_rf(data_dict):
    """
    Train the random forest. 
    """
    # Set up dataset. 
    csv_file = data_dict["csv_file"]
    csv_kwargs = data_dict["csv_kwargs"]
    dataset = VENUS_dataset.Full_VENUS_h5_Dataset(csv_file, csv_kwargs=csv_kwargs)
    # [print(x) for x in dataset.get_cols()]
    run_ids_name = data_dict["run_ids_name"]
    if run_ids_name == "all":
        run_ids = [1,2,3,5,6,6.5,7,7.5,8]
    elif run_ids_name == "first half":
        run_ids = [1,2,3]
    elif run_ids_name == "second half":
        run_ids = [5,6,6.5,7,7.5,8]
    else:
        run_ids = [float(run_ids_name)]
    dataset.filter_value(("run_id", ""), run_ids, mode="in")
    dataset.filter_value(("fcv1_i", "mean"), 1e-5, mode=">")
    # print(run_ids_name, "Length of dataset", len(dataset))

    # Set up column dictionary. 
    cols_dict = {}
    if data_dict["not_input_cols"]:
        not_input_cols = data_dict["not_input_cols"]
        cols_dict["input"] = [
            col for col in dataset.get_cols() if col[0] not in not_input_cols and col[1] != "std"
        ]
    elif data_dict["input_cols"]:
        input_cols = data_dict["input_cols"]
        input_cols = data_dict["input_cols"]
        cols_dict["input"] = [
            col for col in dataset.get_cols() if col[0] in input_cols and col[1] != "std"
        ]
        assert len(cols_dict["input"]) == len(input_cols)
    cols_dict["output"] = [("fcv1_i", "mean")]
    cols_dict["time"] = [("unix_epoch_milliseconds", "mean")]
    # print_list(cols_dict["input"])
    dataset.set_cols(cols_dict)

    val_frac = data_dict["val_frac"]
    input_dim = len(cols_dict["input"])
    
    # Split the training and validation dataset. 
    split_ratios = np.array([1-val_frac, val_frac])
    train_dataset, val_dataset = dataset.random_split(split_ratios)

    # Train the model: Extra-trees regressor. 
    model = sklearn.ensemble.ExtraTreesRegressor()
    # model = sklearn.ensemble.AdaBoostRegressor(model)
    model.fit(train_dataset.get_dataframe("input"), train_dataset.get_dataframe("output") * 1e6)
    score = model.score(val_dataset.get_dataframe("input"), val_dataset.get_dataframe("output") * 1e6)
    pred = model.predict(val_dataset.get_dataframe("input"))
    val_output = val_dataset.get_dataframe("output").to_numpy().squeeze()*1e6
    print(pred.shape, val_dataset.get_dataframe("output").to_numpy().shape)
    print("Score:", score)
    print("Loss", np.mean((pred - val_output)**2))

    # Plot the error vs truth
    plt.plot(val_output, pred-val_output, ".")
    plt.ylabel("Error (uA)")
    plt.xlabel("Actual current (uA)")
    plt.title(f"Validation Error")
    plt.tick_params(direction="in")
    plt.savefig(
        "rf_error.png"
    )
    plt.show()
    plt.close()

    # Plot the feature importance. 
    # print(cols_dict["input"])
    # print(type(model.feature_importances_))
    plt.bar(np.arange(len(model.feature_importances_)), 
        model.feature_importances_,
        label=cols_dict["input"])
    plt.xlabel('Feature Labels')
    plt.xticks(rotation=45)
    plt.ylabel('Feature Importances')
    plt.title('Comparison of different Feature Importances')
    plt.savefig(
        "rf_feature_importance.png"
    )
    plt.show()
    
    # Save the feature importance. 
    feature_importance = pd.DataFrame(
        {
            "Features": cols_dict["input"],
            "Importance": model.feature_importances_
        }
    )
    feature_importance = feature_importance.sort_values("Importance")
    feature_importance.to_csv("rf_feature_importance.csv", index=False)

    
    # Save loss values
    # val_losses = np.array(model_trainer.val_losses).T[1:]
    # train_losses = np.array(model_trainer.train_losses).T
    # val_losses = pd.DataFrame(val_losses, columns=[f"val {loss_fn}" for loss_fn in loss_fns])
    # train_losses = pd.DataFrame(train_losses, columns=[f"train {loss_fn}" for loss_fn in loss_fns])
    # epoch = pd.DataFrame(np.arange(1, num_epochs+1), columns=["epoch"])
    # losses = pd.concat([epoch, val_losses, train_losses], axis=1)
    # losses.to_csv(os.path.join(img_folder, "stable_losses.csv"), index=False)
    
    # Visualize error
    # for err_type in ["abs", "rel"]:
    #     savefig = os.path.join(img_folder, f"stable_{err_type}err_truth.png")
    #     model_trainer.err_plot_truth(["train", "val"], err_type=err_type, 
    #         if_plot=True, savefig=savefig)
    #     savefig = os.path.join(img_folder, f"stable_{err_type}err_time.png")
    #     model_trainer.err_plot_time(["train", "val"], err_type=err_type, 
    #         if_plot=True, savefig=savefig)

    # if data_dict["csv_file_val_data_w_prediction"]:
    #     val_data = next(iter(torch.utils.data.DataLoader(val_dataset, len(val_dataset), shuffle=True)))
    #     pred = trainer.predict(val_data["input"]).detach().flatten().cpu().numpy()
    #     input = val_data["input"].detach().cpu().numpy()
    #     output = val_data["output"].detach().cpu().numpy()
    #     val_data = dataset.recreate_dataframe(input, output, pd.DataFrame(pred, columns=["Prediction"]))
    #     val_data.to_csv(data_dict["csv_file_val_data_w_prediction"], index=False)