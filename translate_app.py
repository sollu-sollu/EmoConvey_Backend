import os

def translate_app():
    # The file we want to translate
    target_file = os.path.join("Emotion-LLaMA", "app_EmotionLlamaClient.py")
    
    if not os.path.exists(target_file):
        print(f"❌ Error: Could not find {target_file}")
        return

    # Dictionary of Chinese phrases and their English translations
    translations = {
        # Comments
        "解析命令行参数": "Parse command line arguments",
        "配置文件路径。": "Configuration file path.",
        "覆盖配置文件中的某些设置，格式为 xxx=yyy。": "Override certain settings in the configuration file, format: xxx=yyy.",
        
        # Error Messages
        "错误：视频文件不存在。": "Error: Video file does not exist.",
        
        # UI Elements
        "视频路径": "Video Path",
        "输入视频文件路径，例如：/path/to/video.mp4": "Enter the video file path, e.g., /path/to/video.mp4",
        "问题": "Question",
        "输入你的问题，例如：视频中的人物表达了什么情绪？": "Enter your question, e.g., What emotion is the person expressing?",
        "模型回答": "Model Output",
        "输入视频路径和问题，Emotion-LLaMA 将解析视频并回答你的问题。": "Input a video path and an optional question. Emotion-LLaMA will analyze the video.",
        
        # UI Titles (If not already changed)
        "Emotion-LLaMA API": "Emotion-LLaMA Local API"
    }

    print(f"📖 Reading {target_file}...")
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Track how many changes were made
    changes_made = 0

    for chinese_text, english_text in translations.items():
        if chinese_text in content:
            content = content.replace(chinese_text, english_text)
            changes_made += 1
            print(f"  ✓ Translated: '{chinese_text}' -> '{english_text}'")

    if changes_made > 0:
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n✅ Successfully replaced {changes_made} Chinese phrases with English in {target_file}!")
    else:
        print("\n⚡ File is already fully translated! No changes needed.")

if __name__ == "__main__":
    translate_app()
