from PIL import Image
import numpy as np
import telebot
import torch


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = checkpoint['model']
    return model


def normalize_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_np = np.array(image)
    image_np = image_np / 255.0
    image_np = (image_np - mean) / std
    image_np = np.clip(image_np, 0, 1)

    return image_np

def predict_pneumonia(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_normalized = normalize_image(image)

    image_tensor = np.transpose(image_normalized, (2, 0, 1))
    image_tensor = np.expand_dims(image_tensor, axis=0)
    image_tensor = torch.FloatTensor(image_tensor)

    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    output = model(image_tensor)
    predicted_class = np.argmax(output.cpu().detach().numpy())

    return predicted_class

model_path = 'checkpoint.pth'


model = load_model(model_path)


API_TOKEN = '6356796877:AAGDnxRzT98iDGISJt6rdVVPhB5XBjFjNZE'


bot = telebot.TeleBot(API_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome_message(message):
    bot.reply_to(message, "Здравствуйте! Отправьте рентгеновский снимок грудной клетки.")


@bot.message_handler(content_types=['photo'])
def handle_messages(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image_path = 'image.jpg'

    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    predicted_class = predict_pneumonia(model, image_path)

    if predicted_class == 1:
        bot.send_message(message.chat.id, "Обнаружена пневмония.")
    else:
        bot.send_message(message.chat.id, "Пневмония не обнаружена.")

bot.polling(none_stop=True)
