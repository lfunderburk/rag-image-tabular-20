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
        """
        Reads JSON content and converts it to Documents, with options for DataFrame flattening and document creation mode.

        :param sources: A list of JSON data sources (file paths or binary objects)
        """
        documents = []

        for source in tqdm(
            sources,
            desc="Converting JSON files to Documents",
            disable=not self.progress_bar,
        ):
            try:
                file_content = self._extract_content(source)
                json_data = json.loads(file_content)

                # Flatten JSON if specified
                if self.flatten_field:
                    df = pd.json_normalize(json_data, record_path=self.flatten_field)
                else:
                    df = pd.DataFrame(json_data if isinstance(json_data, list) else [json_data])

                if self.one_doc_per_row:
                    # Create a document for each row of the DataFrame
                    for _, row in df.iterrows():
                        content = row[self.content_field] if self.content_field and self.content_field in row else None
                        document = Document(content=content, dataframe=pd.DataFrame([row]))
                        documents.append(document)
                else:
                    # Create a single document from the entire DataFrame
                    content = None
                    if self.content_field and self.content_field in df.columns:
                        content = df[self.content_field].iloc[0]

                    document = Document(content=content, dataframe=df)
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
