import pypandoc
import os

md_file = "Arabic_Correction_Journey.md"
docx_file = "Arabic_Correction_Journey.docx"

try:
    pypandoc.convert_file(md_file, 'docx', outputfile=docx_file)
    print(f"Successfully converted {md_file} to {docx_file}")
except Exception as e:
    print(f"Error converting: {e}")

