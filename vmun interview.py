import os
import shutil

# --- Configuration & Paths ---
base_downloads = r"D:\Users\Wilson\Downloads"
source_profiles_dir = os.path.join(base_downloads, "Organized_VMUN_Applications")
target_base_dir = os.path.join(base_downloads, "Director_Interviews")

# Template paths
template_notes = os.path.join(base_downloads, "Director Interview Notes - 2027 Template.docx")
template_backgrounder = os.path.join(base_downloads, "Director Sample Backgrounder - 2027 Template.docx")
template_writing = os.path.join(base_downloads, "Director Writing Sample - 2027 Template.docx")

# First names to filter by (using a set handles the duplicate 'Evan' and 'Sophia' entries)
target_first_names = {
    "Saoirse", "Evan", "Jacqueline", "Nyah", "Juri", "Zack", "Mia", 
    "Kinjal", "Erin", "Anika", "Aziz", "Elysia", "Juno", "Liam", 
    "Sophia", "Holden", "Nima", "Jonathan", "Bella", "Roshin", 
    "Aussia", "Ethan", "Alice", "Marco", "Theo", "Anderson", 
    "Juliet", "Max", "Hassan"
}

def setup_interview_folders():
    # Step 1: Create the large parent folder
    os.makedirs(target_base_dir, exist_ok=True)
    print(f"Target directory verified: {target_base_dir}")

    if not os.path.exists(source_profiles_dir):
        print(f"Error: Could not find source directory at {source_profiles_dir}")
        return

    # Check if templates exist before starting
    for template in [template_notes, template_backgrounder, template_writing]:
        if not os.path.exists(template):
            print(f"Warning: Template not found at {template}. The script will proceed but will skip this file.")

    # Iterate through the organized applications folder
    for folder_name in os.listdir(source_profiles_dir):
        source_folder_path = os.path.join(source_profiles_dir, folder_name)

        # Ensure it's a directory
        if os.path.isdir(source_folder_path):
            # Extract the first name (assuming the format starts with "FirstName LastName")
            parts = folder_name.split()
            if not parts:
                continue
                
            first_name = parts[0]

            if first_name in target_first_names:
                # Extract the full name (e.g., "Aidan Madamba" from "Aidan Madamba - Director")
                full_name = folder_name.split(" - ")[0]
                
                # Set up the new profile folder path in the target directory
                destination_profile_folder = os.path.join(target_base_dir, folder_name)

                # Step 1 (Continued): Copy the entire profile folder
                if not os.path.exists(destination_profile_folder):
                    shutil.copytree(source_folder_path, destination_profile_folder)
                
                # Step 2: Create the nested interview folder
                nested_folder_name = f"{full_name} - D - Interview"
                nested_folder_path = os.path.join(destination_profile_folder, nested_folder_name)
                os.makedirs(nested_folder_path, exist_ok=True)

                # Step 3 & 4: Rename and place the customized word documents
                new_notes_path = os.path.join(nested_folder_path, f"{full_name} - Interview Notes 2027.docx")
                new_backgrounder_path = os.path.join(nested_folder_path, f"{full_name} - Sample Backgrounder 2027.docx")
                new_writing_path = os.path.join(nested_folder_path, f"{full_name} - Director Writing Sample.docx")

                # Copy the templates over with their new names
                if os.path.exists(template_notes):
                    shutil.copy2(template_notes, new_notes_path)
                if os.path.exists(template_backgrounder):
                    shutil.copy2(template_backgrounder, new_backgrounder_path)
                if os.path.exists(template_writing):
                    shutil.copy2(template_writing, new_writing_path)

                print(f"Successfully processed structure for: {full_name}")

    print("\nAll applicant folders have been processed and nested successfully.")

if __name__ == "__main__":
    setup_interview_folders()