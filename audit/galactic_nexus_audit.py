import logging
import datetime
import json
from galactic_nexus_core import GalacticNexusCore
from galactic_nexus_database import GalacticNexusDatabase

class GalacticNexusAudit:
    def __init__(self):
        self.logger = logging.getLogger('galactic_nexus_audit')
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler = logging.FileHandler('galactic_nexus_audit.log')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.database = GalacticNexusDatabase()

    def log_event(self, event_type, event_data, user_id=None, entity_id=None):
        event = {
            'event_type': event_type,
            'event_data': event_data,
            'user_id': user_id,
            'entity_id': entity_id,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.logger.info(json.dumps(event))
        self.database.insert_audit_event(event)

    def log_error(self, error_message, error_code, user_id=None, entity_id=None):
        error = {
            'error_message': error_message,
            'error_code': error_code,
            'user_id': user_id,
            'entity_id': entity_id,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.logger.error(json.dumps(error))
        self.database.insert_audit_error(error)

    def get_audit_events(self, start_date, end_date, user_id=None, entity_id=None):
        query = "SELECT * FROM audit_events WHERE timestamp >= ? AND timestamp <= ?"
        params = (start_date, end_date)
        if user_id:
            query += " AND user_id = ?"
            params += (user_id,)
        if entity_id:
            query += " AND entity_id = ?"
            params += (entity_id,)
        return self.database.retrieve_data('audit_events', query, params)

    def get_audit_errors(self, start_date, end_date, user_id=None, entity_id=None):
        query = "SELECT * FROM audit_errors WHERE timestamp >= ? AND timestamp <= ?"
        params = (start_date, end_date)
        if user_id:
            query += " AND user_id = ?"
            params += (user_id,)
        if entity_id:
            query += " AND entity_id = ?"
            params += (entity_id,)
        return self.database.retrieve_data('audit_errors', query, params)

    def generate_audit_report(self, start_date, end_date, user_id=None, entity_id=None):
        events = self.get_audit_events(start_date, end_date, user_id, entity_id)
        errors = self.get_audit_errors(start_date, end_date, user_id, entity_id)
        report = {
            'events': events,
            'errors': errors
        }
        return report

    def send_audit_report(self, report, recipient_email):
        subject = "Galactic Nexus Audit Report"
        body = "Please find attached the audit report for the specified date range."
        attachment = "audit_report.json"
        with open(attachment, 'w') as f:
            json.dump(report, f)
        GalacticNexusCore.send_email(recipient_email, subject, body, attachment)

galactic_nexus_audit = GalacticNexusAudit()
