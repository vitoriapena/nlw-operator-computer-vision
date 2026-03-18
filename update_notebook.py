import json
import os

filepath = r'c:\workspaces\nlw-operator-computer-vision\webcam_object_detection.ipynb'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# The file has 6 cells: 0 (MD), 1 (CODE !pip), 2 (MD), 3 (CODE download), 4 (MD), 5 (CODE main)

# Update Title
data['cells'][0]['source'] = [
    "# Reconhecimento de Gestos com Webcam, OpenCV e MediaPipe\n",
    "\n",
    "Este notebook utiliza a sua webcam para reconhecer gestos das mãos em tempo real. Ele usa a biblioteca **OpenCV** para capturar os vídeos da câmera e o **MediaPipe Tasks** para processar a imagem e identificar os gestos."
]

# Cell 1 is !pip install, we keep it but can update it to include hands utilities if needed (it already has mediapipe)

# Cell 2: MD for model download
data['cells'][2]['source'] = [
    "### Baixar o modelo de Reconhecimento de Gestos do MediaPipe\n",
    "O MediaPipe Gesture Recognizer requer um arquivo de modelo (`.task`). Vamos baixar o modelo padrão que reconhece gestos como polegar para cima, vitória, etc."
]

# Cell 3: Code for model download
data['cells'][3]['source'] = [
    "import urllib.request\n",
    "import os\n",
    "\n",
    "model_path = 'gesture_recognizer.task'\n",
    "url = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'\n",
    "\n",
    "if not os.path.exists(model_path):\n",
    "    print(\"Baixando o modelo...\")\n",
    "    urllib.request.urlretrieve(url, model_path)\n",
    "    print(\"Modelo baixado com sucesso!\")\n",
    "else:\n",
    "    print(\"O modelo já existe na pasta.\")"
]

# Cell 4: MD for main usage
data['cells'][4]['source'] = [
    "### Reconhecimento de Gestos com a Webcam\n",
    "**Atenção:** Ao rodar a célula abaixo, será aberta uma **nova janela** no seu sistema que acessará a sua câmera.\n",
    "\n",
    "Para fechar a janela, clique nela de forma a ativá-la e aperte a tecla **`q`** ou **`ESC`** no seu teclado. O código irá desenhar os pontos das mãos e o nome do gesto detectado."
]

# Cell 5: Main Code
data['cells'][5]['source'] = [
    "import cv2\n",
    "import mediapipe as mp\n",
    "from mediapipe.tasks import python\n",
    "from mediapipe.tasks.python import vision\n",
    "import numpy as np\n",
    "\n",
    "# 1. Instanciar o reconhecedor de gestos\n",
    "base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')\n",
    "options = vision.GestureRecognizerOptions(base_options=base_options, min_hand_detection_confidence=0.5)\n",
    "recognizer = vision.GestureRecognizer.create_from_options(options)\n",
    "\n",
    "# Ferramentas de desenho do MediaPipe\n",
    "mp_drawing = mp.solutions.drawing_utils\n",
    "mp_hands = mp.solutions.hands\n",
    "mp_drawing_styles = mp.solutions.drawing_styles\n",
    "\n",
    "def draw_landmarks_and_gestures(image, recognition_result):\n",
    "    \"\"\"\n",
    "    Desenha os pontos da mão e o nome do gesto detectado na imagem.\n",
    "    \"\"\"\n",
    "    if not recognition_result.hand_landmarks:\n",
    "        return image\n",
    "\n",
    "    for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):\n",
    "        # Converter landmarks para o formato protobuffer que o draw_landmarks espera\n",
    "        hand_landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()\n",
    "        hand_landmarks_proto.landmark.extend([\n",
    "            mp.framework.formats.landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) \n",
    "            for landmark in hand_landmarks\n",
    "        ])\n",
    "        \n",
    "        # Desenhar os landmarks e conexões\n",
    "        mp_drawing.draw_landmarks(\n",
    "            image,\n",
    "            hand_landmarks_proto,\n",
    "            mp_hands.HAND_CONNECTIONS,\n",
    "            mp_drawing_styles.get_default_hand_landmarks_style(),\n",
    "            mp_drawing_styles.get_default_hand_connections_style())\n",
    "\n",
    "        # Pegar o gesto detectado para esta mão\n",
    "        if recognition_result.gestures and len(recognition_result.gestures) > i:\n",
    "            gesture = recognition_result.gestures[i][0]\n",
    "            gesture_name = gesture.category_name\n",
    "            score = round(gesture.score, 2)\n",
    "            \n",
    "            # Posicionar o texto próximo ao pulso (ponto 0)\n",
    "            wrist = hand_landmarks[0]\n",
    "            h, w, _ = image.shape\n",
    "            text_pos = (int(wrist.x * w), int(wrist.y * h) - 20)\n",
    "            \n",
    "            cv2.putText(image, f\"{gesture_name} ({score})\", text_pos, \n",
    "                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)\n",
    "            \n",
    "    return image\n",
    "\n",
    "# 2. Abrir a Webcam\n",
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "if not cap.isOpened():\n",
    "    print(\"Não foi possível abrir a câmera.\")\n",
    "else:\n",
    "    try:\n",
    "        print(\"Câmera aberta. Pressione 'q' na janela de vídeo para sair.\")\n",
    "        while cap.isOpened():\n",
    "            success, image = cap.read()\n",
    "            if not success:\n",
    "                break\n",
    "\n",
    "            # Espelhar imagem horizontalmente para facilitar a interação\n",
    "            image = cv2.flip(image, 1)\n",
    "            \n",
    "            # Converter BGR para RGB (MediaPipe usa RGB)\n",
    "            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
    "            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)\n",
    "\n",
    "            # Reconhecer o gesto\n",
    "            recognition_result = recognizer.recognize(mp_image)\n",
    "\n",
    "            # Desenhar resultados (pontos e gestos)\n",
    "            annotated_image = draw_landmarks_and_gestures(image, recognition_result)\n",
    "\n",
    "            # Mostrar o resultado\n",
    "            cv2.imshow('MediaPipe Reconhecimento de Gestos', annotated_image)\n",
    "\n",
    "            # Parar se apertarem 'q' ou 'ESC'\n",
    "            key = cv2.waitKey(5) & 0xFF\n",
    "            if key == ord('q') or key == 27:\n",
    "                print(\"Encerrando câmera...\")\n",
    "                break\n",
    "    except Exception as e:\n",
    "        print(f\"Ocorreu um erro: {e}\")\n",
    "    finally:\n",
    "        cap.release()\n",
    "        cv2.destroyAllWindows()\n"
]

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
    print("Notebook atualizado com sucesso!")
