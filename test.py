import json
import os
from typing import Any, Dict
from bot_classifier import BotClassifier
from data_formatter import DataFormatter


def main() -> None:
    formatter = DataFormatter()
    classifier = BotClassifier()
    
    # Cambia esta ruta si quieres usar otro archivo o un string JSON
    json_source = "./sample_input.json"
    
    # Cargar JSON desde archivo o interpretar string JSON
    if os.path.exists(json_source) and os.path.isfile(json_source):
        with open(json_source, 'r', encoding='utf-8') as f:
            raw_data: Dict[str, Any] = json.load(f)
    else:
        raw_data = json.loads(json_source)

    # Procesar los datos y calcular características
    features = formatter.format_data(raw_data)
    result = classifier.predict_detailed(features)
    
    print(f"\n=== RESULTADO DE LA CLASIFICACIÓN ===")
    usuario = raw_data.get('name') or raw_data.get('screen_name') or json_source
    print(f"Usuario: {usuario}")
    print(f"Es humano: {result['prediction']['is_human']}")
    print(f"Confianza: {result['prediction']['confidence']:.2%}")
    print(f"Probabilidad humano: {result['prediction']['probabilities']['human']:.2%}")
    print(f"Probabilidad bot: {result['prediction']['probabilities']['bot']:.2%}")
    
    print(f"\n=== ANÁLISIS DEL MODELO ===")
    print(f"Total de árboles: {result['model_analysis']['total_trees']}")
    print(f"Votos por humano: {result['model_analysis']['human_votes']}")
    print(f"Votos por bot: {result['model_analysis']['bot_votes']}")
    print(f"Ratio de votos: {result['model_analysis']['vote_ratio']:.2%}")
    print(f"Nivel de acuerdo: {result['model_analysis']['agreement_level']:.2%}")
    
    print(f"\n=== CARACTERÍSTICAS MÁS IMPORTANTES ===")
    for i, feature in enumerate(result['feature_importance']['top_features'], 1):
        print(f"{i}. {feature['feature']}: {feature['importance']:.4f}")

    # Mostrar características procesadas
    print(f"\n=== CARACTERÍSTICAS PROCESADAS ===")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
