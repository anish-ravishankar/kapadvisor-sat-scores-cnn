"""
Author: anish.ravishankar@kaplan.com
Date: 2022-01-07
Description: PDF Region Extractor Module
"""

import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output
import warnings
warnings.filterwarnings("ignore")


class PDFRegionExtractor:
    """A class to extract specific regions from PDF documents that contain the score blocks."""
    
    def __init__(self, input_pdf_path: str, output_dir: str, all_subjects: list):
        """
        Initialize the PDFRegionExtractor.
        
        Args:
            input_pdf_path: Path to the input PDF file
            output_dir: Directory to save extracted regions
            all_subjects: List of subject names to extract
        """
        self.input_pdf_path = input_pdf_path
        self.output_dir = output_dir
        self.all_subjects = all_subjects
        self.padding = 10  # Pixels of padding around extracted regions

    def extract_regions(self):
        images = convert_from_path(self.input_pdf_path, dpi=300)
        page_image = images[0]
        ocr_data = pytesseract.image_to_data(page_image, output_type=Output.DICT)
        subject_order = {}

        for subject in self.all_subjects:
            if subject in ocr_data['text']:
                subject_order[subject] = ocr_data['text'].index(subject)
        sorted_subject_order = dict(sorted(subject_order.items(), key=lambda x: x[1]))
        keywords = list(sorted_subject_order.keys())
        keywords_left = list(sorted_subject_order.keys())[:4]
        keywords_right = list(sorted_subject_order.keys())[4:]

        sections = {}
        for i, text in enumerate(ocr_data['text']):
            if text == '' or text == ' ':
                continue
            if text in keywords:
                sections[text] = (ocr_data['left'][i], ocr_data['top'][i])
        height = sections[keywords_left[1]][1] - sections[keywords_left[0]][1]
        for i, text in enumerate(keywords_left):
            x1, y1 = sections[text]
            x2, _ = sections[keywords_right[i]]
            _, y2 = sections[keywords_left[i + 1]] if i < 3 else (0, y1 + height)
            sections[text] = (x1 - self.padding, y1 - self.padding, x2 - self.padding, y2)
            cropped_image = page_image.crop(sections[text])
            self.save_cropped_image(cropped_image, f"{self.output_dir}{text}.png")
    
        width = sections[keywords_left[0]][2] - sections[keywords_left[0]][0]
        for i, text in enumerate(keywords_right):
            x1, y1 = sections[text]
            x2 = x1 + width
            y2 = y1 + height
            sections[text] = (x1 - self.padding, y1 - self.padding, x2 - self.padding, y2)
            cropped_image = page_image.crop(sections[text])
            self.save_cropped_image(cropped_image, f"{self.output_dir}{text}.png")
    
    def save_cropped_image(self, cropped_image, output_path):
        cropped_image.save(output_path)


if __name__ == "__main__":
    # Configuration
    INPUT_PDF_PATH = ('PDFSamples/'
                      'Bluebook PSAT Practice Test_760.pdf')
    OUTPUT_DIR = "CroppedImages/"
    ALL_SUBJECTS = ['Information', 'Algebra', 'Expression', 'Advanced', 
                   'Craft', 'Problem', 'Problem-Solving', 'Standard', 'Geometry']
    

    # Initialize and run extraction
    extractor = PDFRegionExtractor(INPUT_PDF_PATH, OUTPUT_DIR, ALL_SUBJECTS)
    extractor.extract_regions()