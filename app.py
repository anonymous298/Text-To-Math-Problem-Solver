import streamlit as st
from langchain_ollama import ChatOllama
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler

st.title('Text to Math Problem Solver')

llm = ChatOllama(model='llama3.2:3b')

# Initializing the tools
# Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = 'Wikipedia',
    func = wikipedia_wrapper.run,
    description = 'A Tool for searching the internet and solving our math problem'
)

# Initializing Math Tool
math_chain = LLMMathChain()
calculator = Tool(
    name = "Calculator",
    func = math_chain.run,
    description= 'A tool for solving math related problem'
)


prompt = """
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explaination
and display it point wisefor the question below
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    template=prompt,
    input_variables=['question']
)

chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool = Tool(
    name = 'Resoning Tool',
    func = chain.run,
    description='A tool for for answering logic based and reasoning questions'
)

tools = [wikipedia_tool, calculator, reasoning_tool]

agent = initialize_agent(
    tools = tools,
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parsing_errors = True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {'role' : 'assistant', 'content' : 'Hi, Im a Math Assistant Chatbot who can answer all you maths questions'}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

question = st.text_area('Enter your message')

if st.button('find my answer'):
    if question:
        with st.spinner('Generate response..'):
            st.session_state.messages.append({'role' : 'user', 'content' : question})
            st.chat_message('user').write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.invoke(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({'role' : 'assistant', 'content' : response})

            st.write('### Response:')
            st.success(response)

    else:
        st.error("Enter your question please")
