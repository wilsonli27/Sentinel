import os
import shutil

# --- Configuration & Paths ---
# UPDATE THESE PATHS IF THE SCRIPT CANNOT FIND YOUR FOLDERS
base_downloads = r"D:\Users\Wilson\Downloads"
source_profiles_dir = r"D:\Users\Wilson\Downloads\Organized_VMUN_Applications"
target_base_dir = os.path.join(base_downloads, "Targeted_Chair_AD_Interviews")

# Template paths
template_chair_notes = os.path.join(base_downloads, "2026 Chair Interview Notes Template .docx")
template_res_paper = os.path.join(base_downloads, "2026 Resolution Paper Editing Template.docx")
template_crisis_notes = os.path.join(base_downloads, "2026 Crisis Chair Interview Template .docx")
template_crisis_directive = os.path.join(base_downloads, "2026 Crisis Directive Analysis.docx")

# Parsed list of specific applicants
target_full_names = [
    "Alina Somani", "Krupa Pramod", "Sukhmeet Dubb", "Vincent Zhao", "Bowen Du",
    "Aadya Khurana", "Aaria Jamal", "Maximilian Chong", "Valentina Head", "Marie Harmsworth",
    "Cynthia Ma", "Adrian Kim", "Christina Zhang", "Erwin Mok", "Ava Manyal",
    "Joshua Huh", "Jasreen Kaur Johal", "Cindy Zhu", "Noelle McFerran", "Markellan Kaiser",
    "Nathan Lee", "Jeremy Lam", "Keith Alibarbar", "Timothy Liu", "Sam Smith",
    "Christine Wang", "Lily Liu", "Ethan Leung", "Max Wang", "Serena Zhang",
    "Brooklyn Scotland", "Ellaeyn Zhang", "Kevin Zhang", "Kinjal Kaur", "Kaiya Uppal",
    "Alexander Hsiao", "Isabella Wu", "Bonnie Wong", "Eli Juteau", "Ryan Liu",
    "Bernice Ko", "Victoria Bellhouse", "Jayden Sun", "Ashley Wang", "Kaitlyn Kam",
    "Alan Yun", "Anna Zhou", "Charlie Sung", "Gemma Shen", "Yugo Tsao",
    "Yuvakshi Kapoor", "Annie Zhao", "Hukam Kang", "Eva He", "Anria Li",
    "Ariel Liu", "Aileen Jiang", "Annie Lymburner", "Kyle Biemold", "Andy Huang",
    "Xavier Proulx-Sy"
]

# Helper function to normalize names for comparison (removes spaces and makes lowercase)
def normalize_name(name):
    return name.replace(" ", "").lower()

normalized_targets = [normalize_name(name) for name in target_full_names]

def setup_targeted_folders():
    # Step 1: Create the large parent folder
    os.makedirs(target_base_dir, exist_ok=True)
    print(f"Target directory verified: {target_base_dir}")

    if not os.path.exists(source_profiles_dir):
        print(f"\nCRITICAL ERROR: Could not find source directory at:\n{source_profiles_dir}")
        print("Please use 'Copy as path' on your folder and update the 'source_profiles_dir' variable in the script.")
        return

    # Iterate through the organized applications folder
    for folder_name in os.listdir(source_profiles_dir):
        source_folder_path = os.path.join(source_profiles_dir, folder_name)

        if os.path.isdir(source_folder_path):
            # Extract the full name from the folder name
            extracted_full_name = folder_name.split(" - ")[0]
            
            # Check if this applicant is in our specific target list
            if normalize_name(extracted_full_name) in normalized_targets:
                
                # Determine role for folder naming
                is_chair = "Chair" in folder_name
                is_ad = "Assistant Director" in folder_name
                
                # Default to 'C' if neither is explicitly in the title, or map accordingly
                role_indicator = "C" if is_chair else "AD" if is_ad else "Role"
                
                destination_profile_folder = os.path.join(target_base_dir, folder_name)

                # Copy the entire profile folder
                if not os.path.exists(destination_profile_folder):
                    shutil.copytree(source_folder_path, destination_profile_folder)
                
                # Create the nested interview folder
                nested_folder_name = f"{extracted_full_name} - {role_indicator} - Interview"
                nested_folder_path = os.path.join(destination_profile_folder, nested_folder_name)
                os.makedirs(nested_folder_path, exist_ok=True)

                # Rename and place the customized word documents
                new_chair_notes = os.path.join(nested_folder_path, f"{extracted_full_name} - Interview Notes 2027.docx")
                new_res_paper = os.path.join(nested_folder_path, f"{extracted_full_name} - Resolution Paper Editing 2027.docx")
                new_crisis_notes = os.path.join(nested_folder_path, f"{extracted_full_name} - Crisis Interview Notes 2027.docx")
                new_crisis_directive = os.path.join(nested_folder_path, f"{extracted_full_name} - Crisis Directive Analysis 2027.docx")

                # Copy templates over safely
                if os.path.exists(template_chair_notes): shutil.copy2(template_chair_notes, new_chair_notes)
                if os.path.exists(template_res_paper): shutil.copy2(template_res_paper, new_res_paper)
                if os.path.exists(template_crisis_notes): shutil.copy2(template_crisis_notes, new_crisis_notes)
                if os.path.exists(template_crisis_directive): shutil.copy2(template_crisis_directive, new_crisis_directive)

                print(f"Processed: {extracted_full_name}")

    print("\nScript complete. Check the Targeted_Chair_AD_Interviews folder.")

if __name__ == "__main__":
    setup_targeted_folders()