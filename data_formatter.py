from datetime import datetime

class DataFormatter:
    def __init__(self):
        pass

    def format_data(self, raw_data):
        """
        Procesa los datos crudos de Twitter y calcula las características derivadas.
        
        Args:
            raw_data (dict): Diccionario con los datos crudos de la cuenta de la API de Twitter
            
        Returns:
            dict: Características procesadas para el clasificador
        """
        # Convertir valores booleanos a enteros
        description = 1 if raw_data.get('description', '') and raw_data['description'] != '' else 0
        location = 1 if raw_data.get('location', '') and raw_data['location'] != '' else 0
        verified = 1 if raw_data.get('verified', False) else 0
        
        # Obtener valores numéricos
        followers_count = raw_data.get('followers_count', 0)
        friends_count = raw_data.get('friends_count', 0)
        favourites_count = raw_data.get('favourites_count', 0)
        statuses_count = raw_data.get('statuses_count', 0)
        
        # Calcular la edad de la cuenta en días desde created_at
        created_at_str = raw_data.get('created_at', '')
        if created_at_str:
            try:
                # Parsear la fecha ISO 8601
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                account_age_days = (datetime.now(created_at.tzinfo) - created_at).days
            except (ValueError, AttributeError):
                account_age_days = 1  # Valor por defecto si hay error
        else:
            account_age_days = 1
        
        # Calcular tweets promedio por día
        average_tweets_per_day = statuses_count / (account_age_days + 1)
        
        # Calcular características derivadas
        follow_ratio = followers_count / (friends_count + 1)
        popularity_index = (followers_count * favourites_count) / (statuses_count + 1)
        engagement_ratio = favourites_count / (statuses_count + 1)
        activity_index = statuses_count / (account_age_days + 1)
        profile_completeness = description + location + verified
        followers_per_day = followers_count / (account_age_days + 1)
        suspicion_index = followers_count / (statuses_count + 1)
        reciprocity_ratio = friends_count / (followers_count + 1)
        influence_index = (followers_count * verified) / (account_age_days + 1)
        growth_index = followers_count / (account_age_days + 1)
        
        return {
            'description': description,
            'favourites_count': favourites_count,
            'followers_count': followers_count,
            'friends_count': friends_count,
            'location': location,
            'statuses_count': statuses_count,
            'verified': verified,
            'average_tweets_per_day': average_tweets_per_day,
            'account_age_days': account_age_days,
            'follow_ratio': follow_ratio,
            'popularity_index': popularity_index,
            'engagement_ratio': engagement_ratio,
            'activity_index': activity_index,
            'profile_completeness': profile_completeness,
            'followers_per_day': followers_per_day,
            'suspicion_index': suspicion_index,
            'reciprocity_ratio': reciprocity_ratio,
            'influence_index': influence_index,
            'growth_index': growth_index
        }
