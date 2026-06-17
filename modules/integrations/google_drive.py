"""
Google Drive Connector for the Job Hunter Agent

This module provides a simple connector to upload generated resumes, cover letters,
application history, and other files to Google Drive for backup and cross-device access.

Setup Instructions (see also README):
1. Go to https://console.cloud.google.com/
2. Create a new project (or select existing).
3. Enable "Google Drive API".
4. Go to "APIs & Services" > "Credentials" > "Create Credentials" > "OAuth client ID".
5. Application type: "Desktop app".
6. Download the JSON file and save it as `credentials.json` in the project root (or path you configure).
7. The first time you use the connector, it will open a browser for OAuth consent.
   Grant access with the `https://www.googleapis.com/auth/drive.file` scope.
8. A `token.json` (or configured path) will be created for future runs.

Recommended scope: 'https://www.googleapis.com/auth/drive.file'
This allows the app to create and manage only files/folders it creates.

Usage example:
    from modules.integrations.google_drive import get_drive_service, upload_file, upload_job_materials

    service = get_drive_service()
    upload_job_materials("1234567890")   # where 1234567890 is the job_id

    # Or manually:
    upload_file("generated_materials/1234567890/resume_1234567890.pdf", parents=["YOUR_FOLDER_ID"])
"""

import os
import pickle
from typing import Optional, List, Dict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from modules.helpers import print_lg

# Default scopes - least privilege: only files created by this app
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Default paths (relative to project root)
DEFAULT_CREDENTIALS_PATH = "credentials.json"
DEFAULT_TOKEN_PATH = "token.json"


def get_drive_service(
    credentials_path: str = DEFAULT_CREDENTIALS_PATH,
    token_path: str = DEFAULT_TOKEN_PATH,
    scopes: List[str] = None
):
    """
    Authenticates and returns a Google Drive service object.

    Handles token loading, refresh, and interactive OAuth flow on first run.

    Args:
        credentials_path: Path to the OAuth client secrets JSON (from Google Cloud Console).
        token_path: Path where the access/refresh token will be stored.
        scopes: List of Drive API scopes. Defaults to drive.file (recommended).

    Returns:
        googleapiclient.discovery.Resource: Authenticated Drive service.
    """
    if scopes is None:
        scopes = SCOPES

    creds = None

    # Load existing token
    if os.path.exists(token_path):
        try:
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            print_lg(f"Warning: Could not load existing token from {token_path}: {e}")
            creds = None

    # If no valid credentials, run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print_lg("Refreshing Google Drive access token...")
                creds.refresh(Request())
            except Exception as e:
                print_lg(f"Token refresh failed: {e}. Re-authenticating...")
                creds = None

        if not creds:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Google Drive credentials file not found at '{credentials_path}'.\n"
                    "Please download it from Google Cloud Console (APIs & Services > Credentials > OAuth 2.0 Client IDs)\n"
                    "and save it as credentials.json in the project root."
                )

            print_lg("Starting Google Drive OAuth authentication flow...")
            print_lg("A browser window will open. Please log in and authorize the application.")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
            print_lg(f"Google Drive authentication successful. Token saved to {token_path}")

    try:
        service = build('drive', 'v3', credentials=creds)
        print_lg("Google Drive service initialized successfully.")
        return service
    except Exception as e:
        print_lg(f"Failed to build Google Drive service: {e}")
        raise


def create_folder(service, name: str, parent_id: Optional[str] = None) -> str:
    """
    Creates a folder in Google Drive (or returns existing one with same name under parent).

    Args:
        service: Authenticated Drive service.
        name: Folder name.
        parent_id: Optional parent folder ID.

    Returns:
        str: The ID of the created (or existing) folder.
    """
    try:
        # Check if folder already exists
        query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)',
            pageSize=1
        ).execute()

        if results.get('files'):
            folder_id = results['files'][0]['id']
            print_lg(f"Google Drive folder '{name}' already exists (ID: {folder_id}).")
            return folder_id

        # Create new folder
        file_metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]

        folder = service.files().create(
            body=file_metadata,
            fields='id'
        ).execute()

        folder_id = folder.get('id')
        print_lg(f"Created Google Drive folder '{name}' with ID: {folder_id}")
        return folder_id

    except HttpError as error:
        print_lg(f"Error creating Google Drive folder '{name}': {error}")
        raise


def upload_file(
    service,
    local_path: str,
    drive_filename: Optional[str] = None,
    parents: Optional[List[str]] = None,
    mime_type: Optional[str] = None
) -> Optional[Dict]:
    """
    Uploads (or updates if exists with same name in parents) a file to Google Drive.

    Args:
        service: Authenticated Drive service.
        local_path: Path to the local file.
        drive_filename: Name to use in Drive (defaults to local filename).
        parents: List of parent folder IDs.
        mime_type: MIME type (auto-detected if not provided).

    Returns:
        Dict with file metadata (id, name, webViewLink, etc.) or None on failure.
    """
    if not os.path.exists(local_path):
        print_lg(f"Cannot upload: file not found at {local_path}")
        return None

    if drive_filename is None:
        drive_filename = os.path.basename(local_path)

    try:
        # Check if file with same name already exists in the target location
        query = f"name='{drive_filename}' and trashed=false"
        if parents:
            for parent in parents:
                query += f" and '{parent}' in parents"

        existing = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, modifiedTime)',
            pageSize=1
        ).execute()

        file_metadata = {'name': drive_filename}
        if parents:
            file_metadata['parents'] = parents

        media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)

        if existing.get('files'):
            # Update existing file
            file_id = existing['files'][0]['id']
            updated_file = service.files().update(
                fileId=file_id,
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink, modifiedTime'
            ).execute()
            print_lg(f"Updated existing file in Google Drive: {drive_filename} (ID: {updated_file.get('id')})")
            return updated_file
        else:
            # Create new file
            uploaded_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink, webContentLink'
            ).execute()
            print_lg(f"Uploaded new file to Google Drive: {drive_filename} (ID: {uploaded_file.get('id')})")
            return uploaded_file

    except HttpError as error:
        print_lg(f"Google Drive upload failed for {local_path}: {error}")
        return None
    except Exception as e:
        print_lg(f"Unexpected error during Google Drive upload: {e}")
        return None


def upload_job_materials(service, job_id: str, local_base_dir: str = "generated_materials", parent_folder_id: Optional[str] = None) -> List[Dict]:
    """
    Uploads all files for a specific job (from the generated_materials/<job_id> folder) to Google Drive.

    Creates (or reuses) a folder named after the job_id inside the optional parent_folder_id.

    Args:
        service: Authenticated Drive service.
        job_id: The job identifier (folder name under generated_materials).
        local_base_dir: Base directory containing generated materials (default: "generated_materials").
        parent_folder_id: Optional Google Drive folder ID to place the job folder inside.

    Returns:
        List of uploaded file metadata dicts.
    """
    local_job_dir = os.path.join(local_base_dir, str(job_id))
    if not os.path.isdir(local_job_dir):
        print_lg(f"No generated materials folder found for job {job_id} at {local_job_dir}")
        return []

    # Create or get a Drive folder for this job
    job_folder_name = f"Job_{job_id}"
    job_folder_id = create_folder(service, job_folder_name, parent_id=parent_folder_id)

    uploaded_files = []
    for filename in os.listdir(local_job_dir):
        local_path = os.path.join(local_job_dir, filename)
        if os.path.isfile(local_path):
            result = upload_file(
                service,
                local_path=local_path,
                parents=[job_folder_id]
            )
            if result:
                uploaded_files.append(result)

    print_lg(f"Uploaded {len(uploaded_files)} files for job {job_id} to Google Drive.")
    return uploaded_files


def upload_application_history(service, csv_path: str = "all excels/all_applied_applications_history.csv", parent_folder_id: Optional[str] = None) -> Optional[Dict]:
    """
    Uploads (or updates) the main application history CSV to Google Drive.

    Args:
        service: Authenticated Drive service.
        csv_path: Path to the history CSV.
        parent_folder_id: Optional folder to place the file in.

    Returns:
        File metadata dict or None.
    """
    if not os.path.exists(csv_path):
        print_lg(f"Application history CSV not found at {csv_path}")
        return None

    return upload_file(
        service,
        local_path=csv_path,
        drive_filename="linkedin_application_history.csv",
        parents=[parent_folder_id] if parent_folder_id else None
    )


# Convenience function for one-liner usage in runAiBot.py
def upload_after_generation(job_id: str, drive_folder_id: Optional[str] = None, also_upload_history: bool = False) -> bool:
    """
    High-level helper intended to be called after generating tailored materials.

    Args:
        job_id: Current job ID.
        drive_folder_id: Optional top-level Google Drive folder ID to use as root.
        also_upload_history: Whether to also upload the full application history CSV.

    Returns:
        True if at least the job materials were successfully uploaded.
    """
    try:
        service = get_drive_service()
        job_uploads = upload_job_materials(service, job_id, parent_folder_id=drive_folder_id)

        if also_upload_history:
            upload_application_history(service, parent_folder_id=drive_folder_id)

        return len(job_uploads) > 0
    except Exception as e:
        print_lg(f"Google Drive upload after generation failed: {e}")
        return False
