import os
import shutil
import pandas as pd
import re

# 1. File Paths (Matches your provided directories)
pdf_dir = r"D:\Users\Wilson\Downloads\VMUN_2027_Staff_Applicati_May_02_2026_328b"
uploads_base_dir = r"D:\Users\Wilson\Downloads\VMUN2026-05-02-02-26-50\VMUN_2027_Staff_Application_260741380260248"
csv_path = r"VMUN_2027_Staff_Application2026-05-02_02_24_13.csv" # Ensure this is in the same directory as the script

# Where the final organized folders will be created
output_dir = r"D:\Users\Wilson\Downloads\Organized_VMUN_Applications"

# Create the master output directory
os.makedirs(output_dir, exist_ok=True)

# 2. Read the CSV Data
df = pd.read_csv(csv_path)

# 3. Process each applicant
for index, row in df.iterrows():
    full_name = str(row.get('Full Name', '')).strip()
    position = str(row.get('First Choice Position', '')).strip()
    
    # Skip empty rows
    if not full_name or full_name.lower() == 'nan':
        continue
        
    # --- STEP A: Create the applicant's folder ---
    # Strip characters that Windows doesn't allow in folder names
    safe_name = re.sub(r'[\\/*?:"<>|]', "", full_name)
    safe_pos = re.sub(r'[\\/*?:"<>|]', "", position)
    folder_name = f"{safe_name} - {safe_pos}"
    
    applicant_folder = os.path.join(output_dir, folder_name)
    os.makedirs(applicant_folder, exist_ok=True)
    
    # --- STEP B: Find and copy the main PDF ---
    # Attempt to match first and last name to handle dashes like "Aadya-Khurana-12.pdf"
    name_parts = safe_name.split()
    first_name = name_parts[0]
    last_name = name_parts[-1] if len(name_parts) >= 2 else ""
        
    for file in os.listdir(pdf_dir):
        if file.lower().endswith('.pdf') and first_name.lower() in file.lower() and last_name.lower() in file.lower():
            src_pdf = os.path.join(pdf_dir, file)
            shutil.copy(src_pdf, os.path.join(applicant_folder, file))
            break 

    # --- STEP C: Find and copy the Attachments ---
    # The CSV holds the file URLs. We need to extract the 19-digit Submission ID 
    # to find the corresponding local "uploads_" folder.
    resume_url = str(row.get('Please upload your resume.', ''))
    writing_url = str(row.get('Please submit any academic writing sample from the past year.', ''))
    
    # Regex searches for an 18 to 20 digit number (ignoring the 15-digit Form ID)
    id_match = re.search(r'/(\d{18,20})/', resume_url + " " + writing_url)
    
    if id_match:
        submission_id = id_match.group(1)
        upload_folder_name = f"uploads_{submission_id}"
        upload_folder_path = os.path.join(uploads_base_dir, upload_folder_name)
        
        # If the upload folder exists locally, copy all attachments inside
        if os.path.exists(upload_folder_path):
            for attachment in os.listdir(upload_folder_path):
                src_attachment = os.path.join(upload_folder_path, attachment)
                if os.path.isfile(src_attachment):
                    shutil.copy(src_attachment, os.path.join(applicant_folder, attachment))

print(f"Organization complete! Check your new files at {output_dir}")