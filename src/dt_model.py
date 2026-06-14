from model_trainer import ActivityModelTrainer


def main():
    trainer = ActivityModelTrainer(
        data_path="data/cleaned_data.csv",
        model_type="decision_tree",
        model_path="saved_model/decision_tree_model.joblib"
    )

    trainer.run_pipeline()


if __name__ == "__main__":
    main()