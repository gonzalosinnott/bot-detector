#!/usr/bin/env python3
"""
Clasificador simple de bots para uso programático.
Permite clasificar cuentas de Twitter como bot o humano.

Uso:
    from bot_classifier import BotClassifier
    
    classifier = BotClassifier()
    result = classifier.predict({
        'followers_count': 1000,
        'friends_count': 500,
        # ... otras características
    })
"""

import joblib
import pandas as pd

class BotClassifier:
    def __init__(self, model_path='model/random_forest_model.pkl', 
                 feature_columns_path='model/feature_columns.pkl'):
        """
        Inicializa el clasificador.
        
        Args:
            model_path (str): Ruta al modelo entrenado
            feature_columns_path (str): Ruta a las columnas de características
        """
        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(feature_columns_path)
    
    def predict(self, features):
        """
        Clasifica una cuenta como bot o humano.
        
        Args:
            features (dict): Diccionario con las características de la cuenta
            
        Returns:
            dict: {
                'is_human': bool,
                'confidence': float,
                'probabilities': dict
            }
        """
        # Crear DataFrame
        df = pd.DataFrame([features])
        
        # Reordenar columnas
        df = df[self.feature_columns]
        
        # Predecir
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        return {
            'is_human': bool(prediction),
            'confidence': float(max(probabilities)),
            'probabilities': {
                'human': float(probabilities[1]),
                'bot': float(probabilities[0])
            }
        }
    
    def predict_detailed(self, features):
        """
        Clasifica una cuenta como bot o humano con información detallada.
        
        Args:
            features (dict): Diccionario con las características de la cuenta
            
        Returns:
            dict: Información detallada de la clasificación
        """
        # Crear DataFrame
        df = pd.DataFrame([features])
        
        # Reordenar columnas
        df = df[self.feature_columns]
        
        # Predecir
        prediction = self.model.predict(df)[0]
        probabilities = self.model.predict_proba(df)[0]
        
        # Obtener importancia de características
        importance_df = self.get_feature_importance()
        
        # Obtener predicciones de cada árbol individual
        tree_predictions = []
        
        # Obtener el Random Forest del pipeline
        if hasattr(self.model, 'named_steps'):
            # Es un pipeline - necesitamos procesar los datos primero
            rf_model = self.model.named_steps['rf']
            
            # Procesar los datos a través del pipeline hasta el Random Forest
            scaler = self.model.named_steps['scaler']
            pca = self.model.named_steps['pca']
            
            # Escalar los datos
            df_scaled = scaler.transform(df)
            # Aplicar PCA
            df_pca = pca.transform(df_scaled)
            
            # Ahora usar los datos procesados con cada árbol
            for tree in rf_model.estimators_:
                tree_pred = tree.predict(df_pca)[0]
                tree_prob = tree.predict_proba(df_pca)[0]
                tree_predictions.append({
                    'prediction': int(tree_pred),
                    'human_prob': float(tree_prob[1]),
                    'bot_prob': float(tree_prob[0])
                })
        elif hasattr(self.model, 'best_estimator_'):
            # Es un GridSearchCV - obtener el mejor estimador
            best_model = self.model.best_estimator_
            if hasattr(best_model, 'named_steps'):
                # El mejor estimador es un pipeline
                rf_model = best_model.named_steps['rf']
                scaler = best_model.named_steps['scaler']
                pca = best_model.named_steps['pca']
                
                # Procesar los datos
                df_scaled = scaler.transform(df)
                df_pca = pca.transform(df_scaled)
                
                for tree in rf_model.estimators_:
                    tree_pred = tree.predict(df_pca)[0]
                    tree_prob = tree.predict_proba(df_pca)[0]
                    tree_predictions.append({
                        'prediction': int(tree_pred),
                        'human_prob': float(tree_prob[1]),
                        'bot_prob': float(tree_prob[0])
                    })
            else:
                # El mejor estimador es un Random Forest directo
                rf_model = best_model
                for tree in rf_model.estimators_:
                    tree_pred = tree.predict(df)[0]
                    tree_prob = tree.predict_proba(df)[0]
                    tree_predictions.append({
                        'prediction': int(tree_pred),
                        'human_prob': float(tree_prob[1]),
                        'bot_prob': float(tree_prob[0])
                    })
        else:
            # Es un Random Forest directo
            rf_model = self.model
            for tree in rf_model.estimators_:
                tree_pred = tree.predict(df)[0]
                tree_prob = tree.predict_proba(df)[0]
                tree_predictions.append({
                    'prediction': int(tree_pred),
                    'human_prob': float(tree_prob[1]),
                    'bot_prob': float(tree_prob[0])
                })
        
        # Calcular estadísticas de los árboles
        human_votes = sum(1 for tree in tree_predictions if tree['prediction'] == 1)
        bot_votes = len(tree_predictions) - human_votes
        vote_ratio = human_votes / len(tree_predictions)
        
        # Obtener características más importantes para esta predicción
        top_features = importance_df.head(5)
        
        return {
            'prediction': {
                'is_human': bool(prediction),
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'human': float(probabilities[1]),
                    'bot': float(probabilities[0])
                }
            },
            'model_analysis': {
                'total_trees': len(tree_predictions),
                'human_votes': human_votes,
                'bot_votes': bot_votes,
                'vote_ratio': vote_ratio,
                'agreement_level': max(vote_ratio, 1 - vote_ratio)
            },
            'feature_importance': {
                'top_features': top_features.to_dict('records'),
                'all_features': importance_df.to_dict('records')
            },
            'tree_analysis': {
                'individual_predictions': tree_predictions,
                'prediction_variance': float(pd.Series([t['human_prob'] for t in tree_predictions]).var()),
                'prediction_std': float(pd.Series([t['human_prob'] for t in tree_predictions]).std())
            }
        }
    
    def get_feature_importance(self, top_n=None, include_percentages=False):
        """
        Obtiene la importancia de todas las características del modelo.
        
        Args:
            top_n (int, optional): Número de características más importantes a retornar
            include_percentages (bool): Si incluir porcentajes de importancia
            
        Returns:
            pandas.DataFrame: DataFrame con características y su importancia
        """
        # Obtener el Random Forest del pipeline
        if hasattr(self.model, 'named_steps'):
            # Es un pipeline
            rf_model = self.model.named_steps['rf']
        elif hasattr(self.model, 'best_estimator_'):
            # Es un GridSearchCV - obtener el mejor estimador
            best_model = self.model.best_estimator_
            if hasattr(best_model, 'named_steps'):
                # El mejor estimador es un pipeline
                rf_model = best_model.named_steps['rf']
            else:
                # El mejor estimador es un Random Forest directo
                rf_model = best_model
        else:
            # Es un Random Forest directo
            rf_model = self.model
        
        # Obtener las características después de PCA
        if hasattr(self.model, 'named_steps'):
            # Para pipelines con PCA, necesitamos mapear las características originales
            # a los componentes principales
            pca = self.model.named_steps['pca']
            n_components = pca.n_components_
            
            # Crear nombres de características para los componentes PCA
            if isinstance(n_components, float):
                # n_components es un float (porcentaje de varianza)
                n_components = pca.components_.shape[0]
            
            pca_feature_names = [f'PC{i+1}' for i in range(n_components)]
            
            # Obtener importancia de características del Random Forest
            rf_importance = rf_model.feature_importances_
            
            # Mapear de vuelta a las características originales usando los componentes PCA
            # Esto es una aproximación - las importancias reales de las características originales
            # requieren un análisis más complejo
            original_importance = pca.components_.T @ rf_importance
            
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': original_importance
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'best_estimator_') and hasattr(self.model.best_estimator_, 'named_steps'):
            # GridSearchCV con pipeline
            best_model = self.model.best_estimator_
            pca = best_model.named_steps['pca']
            n_components = pca.n_components_
            
            if isinstance(n_components, float):
                n_components = pca.components_.shape[0]
            
            rf_importance = rf_model.feature_importances_
            original_importance = pca.components_.T @ rf_importance
            
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': original_importance
            }).sort_values('importance', ascending=False)
        else:
            # Random Forest directo
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Calcular porcentajes si se solicita
        if include_percentages:
            total_importance = importance_df['importance'].sum()
            importance_df['percentage'] = (importance_df['importance'] / total_importance * 100).round(2)
        
        # Filtrar top N si se especifica
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def get_top_features(self, n=10):
        """
        Obtiene las N características más importantes.
        
        Args:
            n (int): Número de características a retornar
            
        Returns:
            list: Lista de las características más importantes
        """
        importance_df = self.get_feature_importance(top_n=n)
        return importance_df['feature'].tolist()