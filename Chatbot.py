import asyncio
import os
import re
import uuid
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import streamlit as st
import tiktoken
from langchain_community.callbacks import get_openai_callback
from sqlalchemy import create_engine, Table, Column, MetaData, Integer, String, JSON, Text, text
from sqlalchemy.sql import select
from sqlalchemy.dialects.postgresql import ARRAY
import datetime
from datetime import timedelta
from streamlit_tags import st_tags, st_tags_sidebar

from param_summarize_tweet import summarize_every_kol_tweets, summarize_tweet_text, summarize_tweet_text_by_token

os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
<![]()yle>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    engine = create_engine(
        st.secrets["url"]

    )
    return engine


@st.cache_data(ttl=600)
def get_twitter(project_name_list):
    print("刷新数据")

    engine = get_engine()

    # 查询所有的twitter博主
    twitter_list = []
    metadata = MetaData()
    twitter_base_content = Table('twitter_base_content', metadata,
                                 Column('id', Integer, primary_key=True),
                                 Column('tweet_id', String),
                                 Column('influencer_id', String),
                                 Column('original_text', JSON),
                                 Column('publish_time', String)
                                 )
    twitter_base_influencers = Table('twitter_base_influencers', metadata,
                                     Column('id', Integer, primary_key=True),
                                     Column('twitter_username', String),
                                     Column('influencer_id', String),
                                     Column('project_name', String),
                                     Column('project_name_array', ARRAY(Text))
                                     )
    query_project_twitter = select(twitter_base_influencers.c.twitter_username).where(
        twitter_base_influencers.c.project_name_array.op('&&')(project_name_list))
    with engine.connect() as conn:
        if project_name_list and 'daliy_twitter' in project_name_list:
            # query_twitter = select(twitter_base_content.c.influencer_id).group_by(twitter_base_content.c.influencer_id)
            project_name_list = ['daliy_twitter', 'xinsight']
            query_project_twitter = select(twitter_base_influencers.c.twitter_username).where(
                twitter_base_influencers.c.project_name_array.op('&&')(project_name_list))

            result = conn.execute(query_project_twitter)
            for row in result:
                twitter_list.append(row[0])
        else:
            result = conn.execute(query_project_twitter)
            project_twitter_list = []
            for row in result:
                project_twitter_list.append(row[0])
            query_twitter = select(twitter_base_content.c.influencer_id).group_by(
                twitter_base_content.c.influencer_id).having(
                twitter_base_content.c.influencer_id.in_(project_twitter_list))
            result = conn.execute(query_twitter)
            for row in result:
                twitter_list.append(row[0])

    return twitter_list

def get_all_twitter():
    # st.write("You selected:", st.session_state.selected_projects )

    # if not st.session_state.selected_projects:
    #     st.session_state['selection_output'] = []
    twitter_list =  get_twitter(['daliy_twitter'])
    if twitter_list:
        twitter_list.insert(0, 'all')
        st.session_state['selection_output'] = twitter_list
    else:
        st.session_state['selection_output'] = ['no data']
    return twitter_list

with st.sidebar:
    selected_option = st.selectbox('company', ['openai'])
    if selected_option == 'anthropic':
        model_selected_option = st.selectbox('model', ['claude-3-opus-20240229', 'claude-3-sonnet-20240229',
                                                                'claude-3-haiku-20240307'])
    else:
        model_selected_option = st.selectbox('model',
                                             ['gpt-4o'])
    # custom_openai_api_key = st.text_input("API Key", key="chatbot_api_key", type="password")
    custom_openai_api_key = st.secrets["key"]
    if 'selected_projects' not in st.session_state:
        st.session_state['selected_projects'] = []

    # project_options = st.multiselect(
    #     'Please select one or more project',
    #     ['daliy_twitter'],
    #     default=['daliy_twitter'],
    #     key='selected_projects',
    #     on_change=get_all_twitter
    # )
    project_options = ['daliy_twitter']
    get_all_twitter()
    # 设置日期范围的初始值
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=1)

    # 使用st.date_input获取日期范围
    date_range = st.date_input("Select a date range:", [start_date, end_date])

    # st.write("Start date:", date_range[0])
    # st.write("End date:", date_range[1])

    # 获取当前时间
    current_time = datetime.datetime.now().time()
    col1, col2 = st.columns(2)

    # 使用两个st.time_input获取时间范围
    with col1:
        start_time = st.time_input('Select start time:', value=None)
    with col2:
        end_time = st.time_input('Select end time:', value=None)

    # st.write("Start time:", start_time)
    if not start_time:
        start_time = datetime.datetime.now().time()
    if not end_time:
        end_time = datetime.datetime.now().time()
    if len(date_range) > 0:
        start_datetime = datetime.datetime.combine(date_range[0], start_time)
        start_formatted_date = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
    if len(date_range) > 1:
        end_datetime = datetime.datetime.combine(date_range[1], end_time)
        end_formatted_date = end_datetime.strftime('%Y-%m-%d %H:%M:%S')

    if 'selection_output' not in st.session_state:
        st.session_state['selection_output'] = []
    if st.session_state.selection_output:
        options = st.multiselect(
            'Please select one or more twitter',
            st.session_state.selection_output,
        )
    # filter_option = st.selectbox(
    #     "Please select YES or NO filter",
    #     ("YES", "NO"))
    filter_option = 'YES'
    key_words = st_tags_sidebar(
        label='Enter Keywords tag:',
        text='Press enter to add tweet keywords tag',
        suggestions=['btc'],
        maxtags=100
    )
    content_length_limit = st.number_input("Enter length", min_value=0, max_value=10000, step=1,
                                           help='The minimum length of tweet content. Only tweets exceeding this length will be returned.')

    # col32, col33 = st.columns(2)

    # # 使用两个st.time_input获取时间范围
    # with col32:
    #     like_count = st.number_input("Enter like count", min_value=0, max_value=1000000000, step=1,help='The minimum like count of tweet. Only tweets exceeding this count will be returned.')
    # with col33:
    #     quote_count = st.number_input("Enter quote count", min_value=0, max_value=1000000000, step=1,help='The minimum quote count of tweet. Only tweets exceeding this count will be returned.')

    # col34, col35 = st.columns(2)

    # # 使用两个st.time_input获取时间范围
    # with col34:
    #     reply_count = st.number_input("Enter reply count", min_value=0, max_value=1000000000, step=1,help='The minimum reply count of tweet. Only tweets exceeding this count will be returned.')
    # with col35:
    #     retweet_count = st.number_input("Enter retweet count", min_value=0, max_value=1000000000, step=1,help='The minimum retweet count of tweet. Only tweets exceeding this count will be returned.')

    # show_fields = st.multiselect(
    #     'Please select one or more fields',
    #     ['author', 'timestamp', 'source link', 'tweet content'],
    # )
    show_fields = ['author', 'timestamp', 'source link', 'tweet content']

if custom_openai_api_key:
    if selected_option == 'anthropic':
        chat = ChatAnthropic(
            anthropic_api_key=custom_openai_api_key, model_name=model_selected_option, temperature=1)
    else:
        chat = ChatOpenAI(openai_api_key=custom_openai_api_key, model_name=model_selected_option, temperature=1)


def contains_any_efficient(string, char_list):
    """检查字符串是否包含列表中的任一字符或子字符串（不区分大小写）"""
    # 将字符串转换为小写，以便不区分大小写进行比较
    string_lower = string.lower()
    # 遍历列表中的每个字符或子字符串
    for item in char_list:
        # 将列表中的字符或子字符串转换为小写，以便不区分大小写进行比较
        item_lower = item.lower()
        pattern = r'\b{}\b'.format(re.escape(item_lower))
        # 检查小写形式的字符串是否在小写形式的原始字符串中
        if re.search(pattern, string_lower, re.IGNORECASE):
            return True
        # if item_lower in string_lower:
        #     return True
    return False


def all_elements_in_another(list1, list2):
    """检查 list1 的所有元素是否都在 list2 中"""
    return set(list2).issubset(set(list1))


def get_return_tweet(select_return_fields, row):
    if not select_return_fields:
        return f'''author: {row[1]} 
timestamp: {row[3]} 
source link: {row[0]} 
tweet content: {row[2]} {row[4] if row[4] else ''}
-------
'''
    if all_elements_in_another(select_return_fields, ['author', 'timestamp']) or all_elements_in_another(
            select_return_fields, ['author', 'timestamp', 'tweet content']):
        return f'''author: {row[1]} 
timestamp: {row[3]} 
tweet content: {row[2]} {row[4] if row[4] else ''}
-------
'''
    elif all_elements_in_another(select_return_fields, ['author', 'source link', 'tweet content']):
        return f'''author: {row[1]} 
source link: {row[0]} 
tweet content: {row[2]} {row[4] if row[4] else ''}
-------
'''
    elif all_elements_in_another(select_return_fields,
                                 ['timestamp', 'source link', 'tweet content']) or all_elements_in_another(
            select_return_fields, ['timestamp', 'source link']):
        return f'''timestamp: {row[3]} 
source link: {row[0]} 
tweet content: {row[2]} {row[4] if row[4] else ''}
-------
'''
    elif all_elements_in_another(select_return_fields, ['author', 'source link']):
        return f'''author: {row[1]} 
source link: {row[0]} 
tweet content: {row[2]} {row[4] if row[4] else ''} 
-------
'''
    elif all_elements_in_another(select_return_fields, ['author', 'tweet content']):
        return f'''author: {row[1]} 
tweet content: {row[2]} {row[4] if row[4] else ''}
-------
'''
    elif all_elements_in_another(select_return_fields, ['tweet content']):
        return f'''tweet content: {row[2]} {row[4] if row[4] else ''}
-------
'''
    elif all_elements_in_another(select_return_fields, ['author']):
        return f'''author: {row[1]} 
tweet content: {row[2]} {row[4] if row[4] else ''}
-------
'''
    elif all_elements_in_another(select_return_fields, ['source link']):
        return f'''source link: {row[0]} 
tweet content: {row[2]} {row[4] if row[4] else ''}
-------
'''
    elif all_elements_in_another(select_return_fields, ['timestamp']):
        return f'''timestamp: {row[3]} 
tweet content: {row[2]} {row[4] if row[4] else ''}
-------
'''
    else:
        return f'''author: {row[1]} 
timestamp: {row[3]} 
source link: {row[0]} 
tweet content: {row[2]} {row[4] if row[4] else ''}
-------
'''


# def generate_sql(sql):
#     if like_count:
#         sql = sql.concat(" AND like_count > :like_count")
#     if reply_count:
#         sql = sql.concat(" AND reply_count > :reply_count")
#     if quote_count:
#         sql = sql.concat(" AND quote_count > :quote_count")
#     if retweet_count:
#         sql = sql.concat(" AND retweet_count > :retweet_count")
#     return sql

def get_tweet_by_time(is_continue):
    if is_continue:
        return

    total_text = ''
    engine = get_engine()

    # 查询所有的twitter博主
    with engine.connect() as conn:
        if 'all' in options:
            influencer_ids = ", ".join(f"'{elem}'" for elem in st.session_state.selection_output)
            if filter_option == 'YES':
                sql = text(
                    f"select tweet_id, influencer_id,original_text ->> 'text' as tweet_content, publish_time, original_text -> 'quote' ->> 'text' as quote_text from twitter_base_content   where influencer_id in ({influencer_ids}) and trading_opportunity = 1 and publish_time_ts BETWEEN '{str(start_formatted_date)}' AND '{str(end_formatted_date)}' order by influencer_id asc, publish_time_ts desc ")
            else:
                sql = text(
                    f"select tweet_id, influencer_id,original_text ->> 'text' as tweet_content, publish_time, original_text -> 'quote' ->> 'text' as quote_text from twitter_base_content   where influencer_id in ({influencer_ids})  and publish_time_ts BETWEEN '{str(start_formatted_date)}' AND '{str(end_formatted_date)}' order by influencer_id asc, publish_time_ts desc ")

        else:
            influencer_ids = ", ".join(f"'{elem}'" for elem in options)
            if filter_option == 'YES':
                sql = text(
                    f"select tweet_id, influencer_id,original_text ->> 'text' as tweet_content, publish_time, original_text -> 'quote' ->> 'text' as quote_text from twitter_base_content  where influencer_id in ({influencer_ids}) and trading_opportunity = 1 and publish_time_ts BETWEEN '{str(start_formatted_date)}' AND '{str(end_formatted_date)}' order by influencer_id asc, publish_time_ts desc ")
            else:
                sql = text(
                    f"select tweet_id, influencer_id,original_text ->> 'text' as tweet_content, publish_time, original_text -> 'quote' ->> 'text' as quote_text from twitter_base_content  where influencer_id in ({influencer_ids}) and publish_time_ts BETWEEN '{str(start_formatted_date)}' AND '{str(end_formatted_date)}' order by influencer_id asc, publish_time_ts desc")
        # sql = generate_sql(sql)

        result = conn.execute(sql)

        # 根据作者排序，然后再根据时间排序

        for row in result:
            # 判断长度
            if len(str({row[2]}) + str({row[4]})) < content_length_limit and content_length_limit > 0:
                continue
            # 判断是否包含某个字符
            if key_words and not contains_any_efficient((str({row[1]}) + str({row[2]}) + str({row[4]})), key_words):
                continue
            tweet = get_return_tweet(show_fields, row)
            #             tweet = f'''author: {row[1]}
            # timestamp: {row[3]}
            # source link: {row[0]}
            # tweet content: {row[2]} {row[4]}
            # -------
            # '''
            total_text += tweet
    return total_text


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_string(input_string):
    # 检查字符串长度
    if len(input_string) > 50000:
        # 如果超过10000，则截取前10000个字符并加上...
        return input_string[:20000] + '...'
    else:
        # 如果没有超过，返回原字符串
        return input_string


st.title("💬 generate prompt")
display_container0 = st.empty()
display_container = st.empty()
display_container4 = st.empty()
display_container2 = st.empty()
display_container3 = st.empty()


def button_click2():
    is_continue = None
    try:
        if not end_datetime or not start_datetime:
            is_continue = "please select start_time and end_time."
        elif not project_options:
            is_continue = "please select project."
        elif not options:
            is_continue = "please select twitter."

        elif abs(end_datetime - start_datetime) > timedelta(days=1):
            is_continue = "The date interval is more than 1 days."

    except NameError as e:
        is_continue = "please select start_time and end_time."
    data = get_tweet_by_time(is_continue)

    content = 'no data'
    # 根据twitter 和 日期查询推文
    # with st.container(height=500):
    if is_continue:
        content = is_continue
        display_container.empty()

    elif data:
        content = data
        display_container.empty()
    st.session_state['last_content'] = content


def prompt_summit_2():
    if not custom_openai_api_key:
        st.session_state["kol_tweet_output"] = "Please add your OpenAI API key "
    else:
        with st.spinner("processing..."):
            asyncio.run(summarize_tweet_text_by_token(st.session_state.last_content, st.session_state.prompt, chat))

            # data = get_tweet_by_time(None)
            if st.session_state.last_content and len(st.session_state.last_content) > 200:
                # total_result = summarize_tweet_text(st.session_state.last_content,st.session_state.prompt,chat)
                # print('total_result=',total_result)
                if st.session_state.total_result:
                    st.session_state['kol_tweet_output'] = st.session_state.total_result
                else:
                    st.session_state['kol_tweet_output'] = 'no data'


def prompt_summit():
    if not custom_openai_api_key:
        st.session_state["kol_tweet_output"] = "Please add your OpenAI API key "
    else:
        with st.spinner("processing..."):
            # data = get_tweet_by_time(None)
            if st.session_state.last_content and len(st.session_state.last_content) > 200:
                total_result = summarize_tweet_text(st.session_state.last_content, st.session_state.prompt, chat)
                print('total_result=', total_result)
                if total_result:
                    st.session_state['kol_tweet_output'] = total_result
                else:
                    st.session_state['kol_tweet_output'] = 'no data'


def final_prompt_summit():
    final_result = summarize_every_kol_tweets(st.session_state.kol_tweet_output, st.session_state.final_prompt, chat)
    if not final_result or not final_result[0]:
        st.session_state['final_kol_tweet_output'] = 'no data'
    else:
        st.session_state['final_kol_tweet_output'] = final_result[1]


if 'last_content' not in st.session_state:
    st.session_state['last_content'] = ''
if 'kol_tweet_output' not in st.session_state:
    st.session_state['kol_tweet_output'] = ''
if 'final_kol_tweet_output' not in st.session_state:
    st.session_state['final_kol_tweet_output'] = ''
# if st.session_state.last_content:
token_num = num_tokens_from_string(st.session_state.last_content, "cl100k_base")
kol_token_num = num_tokens_from_string(st.session_state.kol_tweet_output, "cl100k_base")
export_file_name = str(uuid.uuid4()) + "_twitter.txt"
export_file_name2 = str(uuid.uuid4()) + "_twitter.txt"
export_file_name3 = str(uuid.uuid4()) + "_twitter.txt"
with display_container0:
    if not st.session_state.last_content or len(st.session_state.last_content) < 100:
        st.markdown(''':rainbow[tweet content]''')
    # elif st.session_state.last_content and len(st.session_state.last_content) >= 100 and (not st.session_state.kol_tweet_output or len(st.session_state.kol_tweet_output) < 50):
    #     col13, col12 = st.columns(2)
    #     with col13:
    #         st.markdown(''':rainbow[tweet content]''')
    #     with col12:
    #         st.markdown(''':rainbow[tweet summrize result]''')
    # elif st.session_state.kol_tweet_output and len(st.session_state.kol_tweet_output) > 50:
    #     col13, col12,col66 = st.columns(3)
    #     with col13:
    #         st.markdown(''':rainbow[tweet content]''')
    #     with col12:
    #         st.markdown(''':rainbow[tweet summrize result]''')
    #     with col66:
    #         st.markdown(''':rainbow[final tweet summrize result]''')
with display_container:
    if not st.session_state.last_content or len(st.session_state.last_content) < 100:
        with st.container(height=500):
            st.code(truncate_string(st.session_state.last_content))
    # elif st.session_state.last_content and len(st.session_state.last_content) >= 100 and (not st.session_state.kol_tweet_output or len(st.session_state.kol_tweet_output) < 50):
    #     col6,col7 = st.columns(2)

    #     with col6:
    #         with st.container(height=500):
    #             st.code(truncate_string(st.session_state.last_content))
    #     with col7:
    #         with st.container(height=500):
    #             st.code(st.session_state.kol_tweet_output)
    # elif st.session_state.kol_tweet_output and len(st.session_state.kol_tweet_output) > 50:
    #     col6,col7,col31 = st.columns(3)
    #     with col6:
    #         with st.container(height=500):
    #             st.code(truncate_string(st.session_state.last_content))
    #     with col7:
    #         with st.container(height=500):
    #             st.code(st.session_state.kol_tweet_output)
    #     with col31:
    #         with st.container(height=500):
    #             st.code(st.session_state.final_kol_tweet_output)
with display_container4:
    if not st.session_state.last_content or len(st.session_state.last_content) < 100:
        st.button("get tweet data", type="primary", on_click=button_click2)
    # elif st.session_state.last_content and len(st.session_state.last_content) >= 100 and (not st.session_state.kol_tweet_output or len(st.session_state.kol_tweet_output) <= 50):
    #     col20, col21 = st.columns(2)
    #     with col20:
    #         st.button("get tweet data", type="primary", on_click=button_click2)
    #     with col21:
    #         st.chat_input(placeholder="please input prompt",on_submit=prompt_summit,key="prompt")
    # elif st.session_state.kol_tweet_output and len(st.session_state.kol_tweet_output)>50:
    #     col20, col21,col23 = st.columns(3)
    #     with col20:
    #         st.button("get tweet data", type="primary", on_click=button_click2)
    #     with col21:
    #         st.chat_input(placeholder="please input prompt",on_submit=prompt_summit,key="prompt")
    #     with col23:
    #         st.chat_input(placeholder="please input prompt",on_submit=final_prompt_summit,key="final_prompt")
with display_container2:
    if not st.session_state.last_content or len(st.session_state.last_content) < 100:
        st.download_button(
            label="export",
            data=st.session_state.last_content,
            file_name=export_file_name,
            mime="text/plain"
        )
    # elif st.session_state.last_content and len(st.session_state.last_content) >= 100 and (not st.session_state.kol_tweet_output or len(st.session_state.kol_tweet_output) < 50):
    #     col3, col4 = st.columns(2)
    #     if st.session_state.last_content:
    #         with col3:
    #             st.download_button(
    #                 label="export",
    #                 data=st.session_state.last_content,
    #                 file_name=export_file_name,
    #                 mime="text/plain"
    #             )
    #     if st.session_state.kol_tweet_output:
    #         with col4:
    #             st.download_button(
    #                 label="export",
    #                 data=st.session_state.kol_tweet_output,
    #                 file_name=export_file_name2,
    #                 mime="text/plain"
    #             )
    # elif st.session_state.kol_tweet_output and len(st.session_state.kol_tweet_output) > 50:
    #     col3, col4,col34 = st.columns(3)
    #     if st.session_state.last_content:
    #         with col3:
    #             st.download_button(
    #                 label="export",
    #                 data=st.session_state.last_content,
    #                 file_name=export_file_name,
    #                 mime="text/plain"
    #             )
    #     if st.session_state.kol_tweet_output:
    #         with col4:
    #             st.download_button(
    #                 label="export",
    #                 data=st.session_state.kol_tweet_output,
    #                 file_name=export_file_name2,
    #                 mime="text/plain"
    #             )
    #     if st.session_state.final_kol_tweet_output:
    #         with col34:
    #             st.download_button(
    #                 label="export",
    #                 data=st.session_state.final_kol_tweet_output,
    #                 file_name=export_file_name3,
    #                 mime="text/plain"
    #             )
# if st.session_state.last_content:
#     with display_container3:
#         st.write('token length = '+ str(token_num))


if st.session_state.last_content and len(st.session_state.last_content) >= 100 and (
        not st.session_state.kol_tweet_output or len(st.session_state.kol_tweet_output) < 50):

    st.markdown(''':rainbow[tweet content]''')
    with st.container(height=500):
        st.code(truncate_string(st.session_state.last_content))
    st.button("get tweet data", type="primary", on_click=button_click2)
    st.download_button(
        label="export",
        data=st.session_state.last_content,
        file_name=export_file_name,
        mime="text/plain"
    )
    if st.session_state.last_content:
        st.write('token length = ' + str(token_num))

    st.write('-----------------------------------------------')

    st.markdown(''':rainbow[tweet summrize result]''')
    with st.container(height=500):
        st.code(st.session_state.kol_tweet_output)
    with st.container(height=80):
        st.chat_input(placeholder="please input prompt", on_submit=prompt_summit_2, key="prompt")
    st.download_button(
        label="export",
        data=st.session_state.kol_tweet_output,
        file_name=export_file_name2,
        mime="text/plain"
    )
    if st.session_state.kol_tweet_output:
        st.write('token length = ' + str(kol_token_num))

if st.session_state.kol_tweet_output and len(st.session_state.kol_tweet_output) > 50:
    st.markdown(''':rainbow[tweet content]''')
    with st.container(height=500):
        st.code(truncate_string(st.session_state.last_content))
    st.button("get tweet data", type="primary", on_click=button_click2)
    st.download_button(
        label="export",
        data=st.session_state.last_content,
        file_name=export_file_name,
        mime="text/plain"
    )
    if st.session_state.last_content:
        st.write('token length = ' + str(token_num))

    st.write('-----------------------------------------------')

    st.markdown(''':rainbow[tweet summrize result]''')
    with st.container(height=500):
        st.code(st.session_state.kol_tweet_output)
    with st.container(height=80):
        st.chat_input(placeholder="please input prompt", on_submit=prompt_summit_2, key="prompt")
    st.download_button(
        label="export",
        data=st.session_state.kol_tweet_output,
        file_name=export_file_name2,
        mime="text/plain"
    )
    if st.session_state.kol_tweet_output:
        st.write('token length = ' + str(kol_token_num))

    st.write('-----------------------------------------------')
    st.markdown(''':rainbow[final tweet summrize result]''')
    with st.container(height=500):
        st.code(st.session_state.final_kol_tweet_output)
    with st.container(height=80):
        st.chat_input(placeholder="please input prompt", on_submit=final_prompt_summit, key="final_prompt")
    st.download_button(
        label="export",
        data=st.session_state.final_kol_tweet_output,
        file_name=export_file_name3,
        mime="text/plain"
    )
