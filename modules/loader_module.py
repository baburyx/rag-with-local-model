from langchain.document_loaders import PDFMinerLoader, WebBaseLoader


class LoaderHandler:
    def __init__(self) -> None:
        pass

    def load_url(self, url):
        data = WebBaseLoader(url).load()
        return data

    def load_pdf(self, pdf_path):
        data = PDFMinerLoader(pdf_path).load()
        return data
