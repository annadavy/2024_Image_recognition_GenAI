from vertexai.preview.vision_models import ImageQnAModel
from vertexai.preview.vision_models import Image
from vertexai.preview.vision_models import ImageGenerationModel
import math
import matplotlib.pyplot as plt




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

def generate_image(prompt,number_of_images):
    
    generation_model = ImageGenerationModel.from_pretrained("imagegeneration@005")
    
    response = generation_model.generate_images(
        prompt=prompt,
        number_of_images=number_of_images
    )

    return response.images[0].show()
    

# An axuillary function to display images in grid
def display_images_in_grid(images):
    """Displays the provided images in a grid format. 4 images per row.

    Args:
        images: A list of PIL Image objects representing the images to display.
    """

    # Determine the number of rows and columns for the grid layout.
    nrows = math.ceil(len(images) / 4)  # Display at most 4 images per row
    ncols = min(len(images) + 1, 4)  # Adjust columns based on the number of images

    # Create a figure and axes for the grid layout.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Display the image in the current axis.
            ax.imshow(images[i]._pil_image)

            # Adjust the axis aspect ratio to maintain image proportions.
            ax.set_aspect("equal")

            # Disable axis ticks for a cleaner appearance.
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Hide empty subplots to avoid displaying blank axes.
            ax.axis("off")

    # Adjust the layout to minimize whitespace between subplots.
    plt.tight_layout()

    # Display the figure with the arranged images.
    plt.show()
    
def generate_and_display_image(prompt,number_of_images):
    
    generation_model = ImageGenerationModel.from_pretrained("imagegeneration@005")
    
    response = generation_model.generate_images(
        prompt=prompt,
        number_of_images=number_of_images
    )

    return display_images_in_grid(response.images)
    