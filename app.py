#########################################
# ЭТАП 1: Конвертация PDF в чанки и изображения
#########################################
import os
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
from typing import Any, Dict, List

# Импорт модулей для конвертации PDF через docling
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.transforms.chunker import HierarchicalChunker


def process_document(document_path: str, output_dir: Path) -> None:
    # Настройки для извлечения структуры PDF и таблиц
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        do_images=True,
        image_extraction_mode="high_quality"
    )
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Настройка чанкинга текста
    chunker = HierarchicalChunker(
        chunk_size=3000,
        chunk_overlap=300,
        split_on_headings=True
    )

    doc_path = Path(document_path)
    doc_result = doc_converter.convert(doc_path)

    if not doc_result or not doc_result.document:
        print(f"Не удалось извлечь текст из {doc_path}")
        return

    # Извлекаем текстовые чанки
    chunks = list(chunker.chunk(doc_result.document))
    processed_chunks = []

    # Директория для сохранения изображений
    img_output_dir = output_dir / "images"
    img_output_dir.mkdir(parents=True, exist_ok=True)

    # Функция для извлечения и сохранения изображений
    def extract_and_save_images(doc_result, img_output_dir: Path) -> None:
        img_output_dir.mkdir(parents=True, exist_ok=True)

        # Предполагаем, что doc_result.document.elements - это список элементов документа
        # Ищем среди них изображения
        for i, element in enumerate(doc_result.document.elements):
            # Предполагаем, что изображения имеют тип элемента "Image"
            # или какой-то специфический атрибут, по которому можно понять, что это изображение.
            # Также предполагаем, что у элемента есть бинарные данные в raw_data или data.

            if getattr(element, 'type', '') == 'Image':
                data = getattr(element, 'raw_data', None) or getattr(element, 'data', None)
                if data:
                    # Определяем расширение по mime-типу, если есть
                    mime_type = getattr(element, 'mime_type', 'image/png')
                    extension = "png"
                    if "jpeg" in mime_type:
                        extension = "jpg"
                    elif "png" in mime_type:
                        extension = "png"
                    elif "gif" in mime_type:
                        extension = "gif"

                    img_filename = f"image_{i + 1}.{extension}"
                    img_path = img_output_dir / img_filename

                    with open(img_path, "wb") as f:
                        f.write(data)

        print(f"Изображения успешно извлечены в директорию {img_output_dir}")

    # Вызов функции сохранения изображений
    extract_and_save_images(doc_result, img_output_dir)

    # Формирование структуры чанков
    for chunk in chunks:
        # Проверяем наличие текста в чанке
        if hasattr(chunk, 'text') and chunk.text and chunk.text.strip():
            # Инициализируем значения по умолчанию
            page_num = None
            headings = []

            # Проверяем, есть ли метаинформация у чанка
            if hasattr(chunk, 'meta'):
                # Извлекаем заголовки, если есть
                if hasattr(chunk.meta, 'headings'):
                    headings = chunk.meta.headings if chunk.meta.headings else []

                # Извлекаем номер страницы, если есть doc_items
                # Предполагается, что chunk.meta.doc_items[0].prov[0].page_no - это способ получения номера страницы
                if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                    first_doc_item = chunk.meta.doc_items[0]
                    if hasattr(first_doc_item, 'prov') and first_doc_item.prov:
                        # Проверяем наличие поля page_no
                        if hasattr(first_doc_item.prov[0], 'page_no'):
                            page_num = first_doc_item.prov[0].page_no

            # Извлекаем изображения для данной страницы, если номер страницы известен
            chunk_images = images_by_page.get(page_num, []) if page_num else []

            # Формируем данные чанка
            chunk_data = {
                'text': chunk.text,
                'metadata': {
                    'headings': headings,
                    'page_number': page_num,
                    'images': chunk_images
                }
            }

            # Добавляем сформированный чанк в общий список
            processed_chunks.append(chunk_data)

    # Сохраняем результаты этапа 1
    output_path = output_dir / f"{doc_path.stem}_processed.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'filename': doc_path.name,
            'chunks': processed_chunks,
            'total_chunks': len(processed_chunks)
        }, f, ensure_ascii=False, indent=2)

    print(f"Обработан документ: {doc_path.name}, Чанков: {len(processed_chunks)}")

def main_stage1():
    data_dir = Path("data")  # Директория, где лежат ваши PDF
    output_dir = Path("processed_results_stage1")
    output_dir.mkdir(exist_ok=True)

    pdf_files = list(data_dir.glob("*.pdf"))
    print(f"Найдено PDF файлов: {len(pdf_files)}")

    for pdf_path in pdf_files:
        print(f"\nОбработка {pdf_path.name}")
        process_document(str(pdf_path), output_dir)

#########################################
# ЭТАП 2: Построение эмбеддингов для текстов и изображений
#########################################

import torch
import base64
import requests
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


# Настройки для модели Qwen2-VL
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
vl_model.eval()

min_pixels = 448 * 448
max_pixels = 896 * 896
vl_model_processor = Qwen2VLProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

def get_text_embeddings(texts: List[str]) -> torch.Tensor:
    text_inputs = vl_model_processor(text=texts, padding=True, return_tensors="pt").to(vl_model.device)

    with torch.no_grad():
        outputs = vl_model(
            **text_inputs,
            output_hidden_states=True,
            return_dict=True
        )
    last_hidden_states = outputs.hidden_states[-1]

    attention_mask = text_inputs['attention_mask']
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * expanded_mask, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings.detach().cpu()

def get_image_embeddings_from_file(image_path: str) -> torch.Tensor:
    with open(image_path, "rb") as img_f:
        image_bytes = img_f.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    image_format = image.format.lower() if image.format else "jpeg"
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:image/{image_format};base64,{base64_image}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": data_url
                },
                {
                    "type": "text",
                    "text": "Опишите содержимое этого изображения: <|image|>"
                },
            ],
        }
    ]

    text = vl_model_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = vl_model_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(vl_model.device)

    with torch.no_grad():
        outputs = vl_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    last_hidden_states = outputs.hidden_states[-1]
    image_token_id = vl_model.config.image_token_id
    image_token_mask = (inputs["input_ids"] == image_token_id)
    image_token_count = image_token_mask.sum(dim=1, keepdim=True).clamp_min(1e-9)
    image_embeddings = (last_hidden_states * image_token_mask.unsqueeze(-1)).sum(dim=1) / image_token_count

    return image_embeddings.detach().cpu()

def main_stage2():
    input_dir = Path("processed_results_stage1")  # Результаты этапа 1
    output_dir = Path("processed_results_final")
    output_dir.mkdir(exist_ok=True)

    for json_file in input_dir.glob("*_processed.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = data['chunks']

        # Получаем текстовые эмбеддинги для всех чанков
        all_texts = [ch['text'] for ch in chunks]
        text_embeddings = get_text_embeddings(all_texts)

        # Добавляем эмбеддинги к чанкам
        for i, ch in enumerate(chunks):
            ch['text_embedding'] = text_embeddings[i].tolist()

            # Получаем эмбеддинги для изображений
            image_embeddings = []
            for img_path in ch['metadata']['images']:
                emb = get_image_embeddings_from_file(img_path)
                image_embeddings.append(emb[0].tolist())  # берем первый вектор
            ch['image_embeddings'] = image_embeddings

        # Сохраняем обновлённую структуру с эмбеддингами
        data['chunks'] = chunks
        output_path = output_dir / json_file.name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Эмбеддинги добавлены: {json_file.name}")

##################################
# Запуск по необходимости:
# Сначала выполнить этап 1, затем этап 2.
##################################

if __name__ == "__main__":
    # Сначала этап 1:
    main_stage1()
    # После успешного выполнения этапа 1,
    # можно выполнить этап 2 (при необходимости):
    # main_stage2()
