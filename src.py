from vertexai.preview.vision_models import ImageQnAModel
from vertexai.preview.vision_models import Image


def show_image(image):
    image_load = Image.load_from_file(image)
    image_load.show()
    
def ask_image(image, question, number_of_results):
    
    image_qna_model = ImageQnAModel.from_pretrained("imagetext@001")
    image_load = Image.load_from_file(image)
    
    return image_qna_model.ask_question(
        image=image_load, question=question, number_of_results=number_of_results)

def caption_image(image, number_of_results, language):
   
    image_captioning_model = ImageCaptioningModel.from_pretrained("imagetext@001")
    
    image=Image.load_from_file(image)
    
    return image_captioning_model.get_captions(
        image=image,
        number_of_results=number_of_results,
        language=language)
    