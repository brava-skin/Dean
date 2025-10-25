
# Fallback handlers for non-existent Supabase tables
def safe_supabase_insert(client, table_name, data):
    """Safely insert data, ignoring errors for non-existent tables"""
    try:
        return client.table(table_name).insert(data).execute()
    except Exception as e:
        if "Could not find the table" in str(e):
            # Table doesn't exist, silently ignore
            return None
        else:
            # Re-raise other errors
            raise e

def safe_supabase_upsert(client, table_name, data, on_conflict=None):
    """Safely upsert data, ignoring errors for non-existent tables"""
    try:
        if on_conflict:
            return client.table(table_name).upsert(data, on_conflict=on_conflict).execute()
        else:
            return client.table(table_name).upsert(data).execute()
    except Exception as e:
        if "Could not find the table" in str(e):
            # Table doesn't exist, silently ignore
            return None
        else:
            # Re-raise other errors
            raise e
