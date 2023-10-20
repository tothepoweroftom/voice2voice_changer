# voice2voice_changer
This project gonna help you could inference and train RVC model for voice to voice changer real-time application

Modules in this folder from https://github.com/w-okada/voice-changer 
I have added a inference file to convert voice with available sample model
## How to run it?
Step 1: Clone this repo
- Use command:
```
  git clone https://github.com/ndturng/voice2voice_changer
```
Step 2: Create enviroment, activate it and install requirements, download the sample models and weights
- Create an enviroment, activate it
- Install the libraries in requirements.txt file, use this command
```
  pip install -r requirements.txt
```
- Download the sample models and weights
```
  python3 my_inference.py
```
  The ```my_inference.py``` will auto download models and weights if it has not been downloaded yet 
  
Step 3: Modify in my_inference.py

3.1 Input and output file path
```
  audio_path = "your/path/input_file_name.wav"
  output_path = "your/path/output_file_name.wav"
```
3.2 Modify slotInfo
```
  slotInfo: RVCModelSlot = RVCModelSlot(
        slotIndex=0,
        modelFile="tsukuyomi_v2_40k_e100_simple.onnx",
        modelType="onnxRVC",
        isONNX=True
    )
```
+ There are 6 models, ```slotIndex=0``` for model in model_dir/0
+ Go inside the folder model_dir/0 and copy the model name ```tsukuyomi_v2_40k_e100_simple.onnx```, modify it at the line
```
  modelFile="tsukuyomi_v2_40k_e100_simple.onnx"
```
3.3 Update tune
Num 22 for male voice convert to female voice, use negative for the opposite, range -25 to 25
```
  voiceChangerManager.update_settings("tran", 22)
```
Step 4: Run my_inference.py file
```
  python3 my_inference.py
```
NOTE: If there's a bug with ```portaudio19```, use this command:
```
  sudo apt-get install portaudio19-dev python-all-dev
```
