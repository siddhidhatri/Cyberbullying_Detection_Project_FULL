import os
from scripts.train_model import train

def launch_app():
    print("🚀 Launching Streamlit app...")
    os.system("streamlit run scripts/app.py")

def main():
    print("🎯 Cyberbullying Detection System")
    print("Choose an option:")
    print("1 - Train the model")
    print("2 - Run the Streamlit app")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        train()
    elif choice == "2":
        launch_app()
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
