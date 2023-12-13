import json
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from haystack import Document, component
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)


@component
class JsonToDocument:
    """
    Converts JSON content into Haystack Documents, with options to flatten JSON and create either a single document 
    from the entire table or one document per row.

    Usage example:
    ```python
    from your_module import JsonToDocument  # Replace with actual module name

    converter = JsonToDocument(content_field="title", flatten_field="data", one_doc_per_row=True)
    results = converter.run(sources=["complex_json.json"])
    documents = results["documents"]
    # Each document represents a row if one_doc_per_row is True
    ```
    """

    def __init__(self, content_field: str = None, flatten_field: str = None, one_doc_per_row: bool = False, progress_bar: bool = True):
        """
        :param content_field: The field in JSON to use as the content of the Document. Set to None if not needed. Defaults to None.
        :param flatten_field: The field in JSON to flatten using json_normalize. Set to None if not needed. Defaults to None.
        :param one_doc_per_row: Whether to create a separate document for each row of the DataFrame. Defaults to False.
        :param progress_bar: Show a progress bar for the conversion. Defaults to True.
        """
        self.content_field = content_field
        self.flatten_field = flatten_field
        self.one_doc_per_row = one_doc_per_row
        self.progress_bar = progress_bar

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]]):
        documents = []

        for source in tqdm(sources, desc="Converting JSON files to Documents", disable=not self.progress_bar):
            try:
                file_content = self._extract_content(source)
                json_data = json.loads(file_content)

                # Determine if json_data is a list or a single dict
                if isinstance(json_data, dict):
                    json_data = [json_data]  # Make it a list for uniform processing

                # Create documents based on the user's choice
                if self.one_doc_per_row:
                    # Create one document per JSON object
                    for item in json_data:
                        content = item.get(self.content_field, None) if self.content_field else None
                        document = Document(content=content, meta=item)
                        documents.append(document)
                else:
                    # Create a single document from all JSON objects
                    content = None
                    combined_meta = {}
                    for item in json_data:
                        combined_meta.update(item)  # Merge all items into a single meta
                        if self.content_field and self.content_field in item:
                            content = item[self.content_field]  # Set content from the last item
                    document = Document(content=content, meta=combined_meta)
                    documents.append(document)

            except Exception as e:
                logger.warning("Failed to process %s. Error: %s", source, e)

        return {"documents": documents}



    def _extract_content(self, source: Union[str, Path, ByteStream]) -> str:
        """
        Extracts content from the given data source.
        :param source: The data source to extract content from.
        :return: The extracted content.
        """
        if isinstance(source, (str, Path)):
            with open(source, 'r', encoding='utf-8') as file:
                return file.read()
        if isinstance(source, ByteStream):
            return source.data.decode('utf-8')

        raise ValueError(f"Unsupported source type: {type(source)}")
