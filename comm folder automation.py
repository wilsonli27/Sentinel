import os
import shutil

def setup_vmun_folders(base_dir, committees, files_to_keep, files_to_rename):
    """
    Creates committee folders and copies/renames template files into them.
    """
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return

    print(f"\nProcessing directory: {base_dir}")
    
    for committee in committees:
        committee_dir = os.path.join(base_dir, committee)
        
        # 1. Create the committee folder
        os.makedirs(committee_dir, exist_ok=True)
        print(f"Created folder: {committee}")

        # 2. Copy the PDF files that don't need a name change
        for file in files_to_keep:
            src = os.path.join(base_dir, file)
            dst = os.path.join(committee_dir, file)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"  -> Missing source file to keep: {file}")

        # 3. Copy and rename the Word files based on the committee name
        for original_name, new_name_pattern in files_to_rename.items():
            src = os.path.join(base_dir, original_name)
            
            new_filename = new_name_pattern.format(Committee=committee)
            dst = os.path.join(committee_dir, new_filename)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"  -> Missing source file to rename: {original_name}")

if __name__ == "__main__":
    
    # ==========================================
    # PHASE 1: General Committees
    # ==========================================
    comm_base_dir = r"D:\Users\Wilson\Downloads\commfoldertemplate"
    comm_committees = ["UNHRC", "UNESCO", "IMF", "UNEP", "ISA", "UNODC", "NATO", "OAS", "EU", "HOC", "UNSC"]
    
    # Updated to .pdf and exact string from your screenshot
    comm_files_to_keep = [
        "Sample Background Guide.pdf", 
        "VMUN 2027 - Director's Guide.pdf"
    ]
    
    # Updated to match the exact .docx file names in your folder
    comm_files_to_rename = {
        "Topic A.docx": "{Committee} - Topic A.docx",
        "Topic B.docx": "{Committee} - Topic B.docx",
        "Director's Letter.docx": "{Committee} - Director's Letter.docx",
        "Committee Description.docx": "{Committee} - Committee Description.docx"
    }

    setup_vmun_folders(comm_base_dir, comm_committees, comm_files_to_keep, comm_files_to_rename)


    # ==========================================
    # PHASE 2: Crisis Committees
    # ==========================================
    crisis_base_dir = r"D:\Users\Wilson\Downloads\crisistemplate"
    crisis_committees = ["INTEL", "Cabinet", "HCC", "ACC"]
    
    # Assuming the Crisis background guide is also a PDF based on the other folder
    crisis_files_to_keep = [
        "Sample Crisis Background Guide.pdf", 
        "VMUN 2027 - Director's Guide.pdf"
    ]
    
    # Updated to match the expected exact .docx file names
    crisis_files_to_rename = {
        "Crisis BG Template.docx": "{Committee} - Crisis BG.docx",
        "Director's Letter.docx": "{Committee} - Director's Letter.docx",
        "Committee Description.docx": "{Committee} - Committee Description.docx"
    }

    setup_vmun_folders(crisis_base_dir, crisis_committees, crisis_files_to_keep, crisis_files_to_rename)
    
    print("\nFolder structuring complete.")