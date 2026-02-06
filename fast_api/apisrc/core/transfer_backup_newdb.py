# migrate_data.py
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.orm import sessionmaker

# Database URLs
OLD_DB_URL = "postgresql://enpower:enpower@147.102.6.183:5432/scot"  
NEW_DB_URL = "postgresql://enpower:enpower@147.102.6.183:5432/scot_final"

def migrate_data():
    # Connect to both databases
    old_engine = create_engine(OLD_DB_URL)
    new_engine = create_engine(NEW_DB_URL)
    
    OldSession = sessionmaker(bind=old_engine)
    NewSession = sessionmaker(bind=new_engine)
    
    old_session = OldSession()
    new_session = NewSession()
    
    # Reflect old database structure
    old_metadata = MetaData()
    old_metadata.reflect(bind=old_engine)
    
    try:
        # Migrate Sites
        old_sites = old_metadata.tables['sites']
        old_site_data = old_session.execute(old_sites.select()).fetchall()
        
        print(f"Migrating {len(old_site_data)} sites...")
        for row in old_site_data:
            insert_data = {
                'id': row.id,
                'name': row.name,
                'latitude': 36.2296,
                'longitude': 27.5672
            }
            new_session.execute(
                text("INSERT INTO sites (id, name, latitude, longitude) VALUES (:id, :name, :latitude, :longitude)"),
                insert_data
            )
        
        # Migrate ConsumptionData
        old_consumption = old_metadata.tables['consumption_data']
        old_consumption_data = old_session.execute(old_consumption.select()).fetchall()

        old_production = old_metadata.tables['production_data']
        old_production_data = old_session.execute(old_production.select()).fetchall()
        
        old_forecasted_production = old_metadata.tables['forecasted_production_data']
        old_forecasted_production_data = old_session.execute(old_forecasted_production.select()).fetchall()

        old_forecasted_consumption = old_metadata.tables['forecasted_consumption_data']
        old_forecasted_consumption_data = old_session.execute(old_forecasted_consumption.select()).fetchall()



        print(f"Migrating {len(old_consumption_data)} consumption records...")
        for row in old_consumption_data:
            new_session.execute(
                text("INSERT INTO consumption_data (site_id, timestamp, value) VALUES (:site_id, :timestamp, :value)"),
                {'site_id': row.site_id, 'timestamp': row.timestamp, 'value': row.value}
            )
        for row in old_production_data:
            new_session.execute(
                text("INSERT INTO production_data (site_id, timestamp, value) VALUES (:site_id, :timestamp, :value)"),
                {'site_id': row.site_id, 'timestamp': row.timestamp, 'value': row.value}
            )
        for row in old_forecasted_consumption_data:
            new_session.execute(
                text("INSERT INTO forecasted_consumption_data (site_id, timestamp, value) VALUES (:site_id, :timestamp, :value)"),
                {'site_id': row.site_id, 'timestamp': row.timestamp, 'value': row.value}
            )
        for row in old_forecasted_production_data:
            new_session.execute(
                text("INSERT INTO forecasted_production_data (site_id, timestamp, value) VALUES (:site_id, :timestamp, :value)"),
                {'site_id': row.site_id, 'timestamp': row.timestamp, 'value': row.value}
            )
        new_session.commit()
        print("Migration completed successfully!")
        
    except Exception as e:
        new_session.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        old_session.close()
        new_session.close()

if __name__ == "__main__":
    migrate_data()