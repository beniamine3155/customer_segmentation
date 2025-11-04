import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.constants import SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            self.scaler = StandardScaler()
        except Exception as e:
            raise MyException(e, sys)
        
    
    def transform_customer_segmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations specific to customer segmentation.
        """
        # Drop missing values
        df = df.dropna()

        # Convert Dt_Customer to datetime
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)

        # Create Age
        df['Age'] = 2025 - df['Year_Birth']

        # Total children
        df['Total_Children'] = df['Kidhome'] + df['Teenhome']

        # Total spend
        spend_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                         'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        df['Total_Spend'] = df[spend_columns].sum(axis=1)

        # Customer since days
        df['Customer_Since_Days'] = (pd.to_datetime('today') - df['Dt_Customer']).dt.days

        # Accepted any campaign
        df['AcceptedAny'] = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                                'AcceptedCmp4','AcceptedCmp5','Response']].sum(axis=1)
        df['AcceptedAny'] = df['AcceptedAny'].apply(lambda x: 1 if x > 0 else 0)

        # Age groups
        bins = [18, 30, 40, 50, 60, 70, 90]
        labels = ['18-29','30-39','40-49','50-59','60-69','70+']
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

        # Scale selected features
        features = ['Age', 'Income', 'Total_Spend', 'Recency',
                    'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
        df_scaled = df.copy()
        df_scaled[features] = self.scaler.fit_transform(df_scaled[features])

        return df_scaled, features

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation process")

            # Load data
            input_file_path = self.data_ingestion_artifact.feature_store_file_path
            df = pd.read_csv(input_file_path)
            logging.info(f"Loaded data from {input_file_path} with shape {df.shape}")

            # Transform data
            transformed_df, feature_columns = self.transform_customer_segmentation(df)
            logging.info("Data transformation completed")

            # Create directories
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.transformed_object_dir, exist_ok=True)

            # Save transformed data
            transformed_df.to_csv(self.data_transformation_config.transformed_data_path, index=False)
            logging.info(f"Transformed data saved to {self.data_transformation_config.transformed_data_path}")

            # Save scaler object
            save_object(self.data_transformation_config.scaler_object_path, self.scaler)
            logging.info(f"Scaler object saved to {self.data_transformation_config.scaler_object_path}")

            # Return artifact
            return DataTransformationArtifact(
                transformed_data_path=self.data_transformation_config.transformed_data_path,
                scaler_object_path=self.data_transformation_config.scaler_object_path,
                feature_columns=feature_columns
            )
        except Exception as e:
            raise MyException(e, sys)