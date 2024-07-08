
from dataclasses import dataclass, field

from lightrag.core import Component, Generator, DataClass
from lightrag.components.model_client import GroqAPIClient
from lightrag.components.output_parsers import JsonOutputParser

"""We dont need dataclass as of now will setup if needed in future"""

# we might need it lol

@dataclass
class ArxivSummaryOutput(DataClass):
    brief: str = field(
        metadata={"desc": "A brief explaination of the concept introduced in one sentence."}
    )
    potential_applications: str = field(metadata={"desc": "Potential aplications of concept/ study introduced in the paper"})



qa_template = r"""<SYS>
You are a helpful assistant, you are given a research paper title and its summary. 
<OUTPUT_FORMAT>
{{output_format_str}}
</OUTPUT_FORMAT>
</SYS>
User: {{input_str}}
You:"""



class ArxivQA(Component):
    def __init__(self):
        super().__init__()

        parser = JsonOutputParser(data_class=ArxivSummaryOutput, return_data_class=True)
        self.generator = Generator(
            model_client=GroqAPIClient(),
            model_kwargs={"model": "llama3-8b-8192"},
            template=qa_template,
            prompt_kwargs={"output_format_str": parser.format_instructions()},
            output_processors=parser,
        )

    def call(self, query: str):
        return self.generator.call({"input_str": query})

    async def acall(self, query: str):
        return await self.generator.acall({"input_str": query})
