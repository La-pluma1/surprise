import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# 获取当前脚本所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_trained_model():
    """加载Surprise训练的模型（.pkl格式）"""
    model_path = os.path.join(current_dir, 'svd_model.pkl')
    st.write(f"尝试加载模型: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("Surprise模型加载成功！")
        return model
    except FileNotFoundError:
        st.error(f"模型文件不存在: {model_path}")
        st.error(f"请确认模型文件是否存在于: {current_dir}")
        st.error(f"当前目录内容: {os.listdir(current_dir)}")
        return None
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

@st.cache_data
def load_experiment_data(ratings_path='processed_ratings.csv', jokes_path='Dataset4JokeSet.xlsx'):
    """加载评分数据和笑话文本，增加文件存在检查和编码处理"""
    # 构建绝对路径
    full_ratings_path = os.path.join(current_dir, ratings_path)
    full_jokes_path = os.path.join(current_dir, jokes_path)
    
    st.write(f"尝试加载评分数据: {full_ratings_path}")
    st.write(f"尝试加载笑话数据: {full_jokes_path}")
    
    # 检查文件是否存在
    if not os.path.exists(full_ratings_path):
        st.error(f"评分文件不存在: {full_ratings_path}")
        st.error(f"请确认文件是否存在于: {current_dir}")
        st.error(f"当前目录内容: {os.listdir(current_dir)}")
        raise FileNotFoundError(f"找不到文件: {full_ratings_path}")
    
    if not os.path.exists(full_jokes_path):
        st.error(f"笑话文件不存在: {full_jokes_path}")
        st.error(f"请确认文件是否存在于: {current_dir}")
        st.error(f"当前目录内容: {os.listdir(current_dir)}")
        raise FileNotFoundError(f"找不到文件: {full_jokes_path}")
    
    try:
        # 尝试多种编码读取CSV文件
        try:
            ratings_df = pd.read_csv(full_ratings_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                ratings_df = pd.read_csv(full_ratings_path, encoding='gbk')
            except UnicodeDecodeError:
                ratings_df = pd.read_csv(full_ratings_path, encoding='ISO-8859-1')
        
        st.success("评分数据加载成功！")
    except Exception as e:
        st.error(f"读取评分数据失败: {e}")
        raise
    
    try:
        # 读取笑话数据，假设Excel中只有笑话文本
        jokes_df = pd.read_excel(full_jokes_path, header=None, names=['joke_text'])
        # 为笑话添加ID
        jokes_df['joke_id'] = jokes_df.index + 1
        st.success("笑话数据加载成功！")
    except Exception as e:
        st.error(f"读取笑话数据失败: {e}")
        raise
    
    return ratings_df, jokes_df

def display_random_jokes(jokes_df):
    """随机展示3个笑话并收集评分（-10到10）"""
    # 初始化session_state，确保selected_jokes和user_ratings存在
    if 'selected_jokes' not in st.session_state:
        joke_ids = jokes_df['joke_id'].unique()
        st.session_state.selected_jokes = random.sample(list(joke_ids), 3)  # 随机选3个ID
        st.session_state.user_ratings = {}  # 存储评分

    st.header("请为以下笑话评分（-10到10）")
    cols = st.columns(3)  # 三列布局

    for i, joke_id in enumerate(st.session_state.selected_jokes):
        with cols[i]:
            # 获取当前笑话文本
            joke_text = jokes_df[jokes_df['joke_id'] == joke_id]['joke_text'].iloc[0]
            st.subheader(f"笑话 {i+1}")
            st.write(joke_text)
            
            # 关键修改：给slider添加唯一key，用joke_id保证唯一性
            rating = st.slider(
                f"评分", 
                min_value=-10.0, 
                max_value=10.0, 
                value=0.0,        # 默认评分为0
                step=0.1,         # 评分步长
                key=f"slider_rating_{joke_id}"  # 唯一key，避免ID冲突
            )
            
            # 将评分存入session_state
            st.session_state.user_ratings[joke_id] = rating  

    # 提交评分按钮
    if st.button("提交评分"):
        st.success("评分已提交！")
        # 展示用户提交的评分
        st.write("您的评分:", st.session_state.user_ratings)

def generate_surprise_recommendations(user_ratings, model, ratings_df, jokes_df, top_n=5):
    """使用Surprise模型生成推荐"""
    # 生成新用户ID（使用一个不存在的ID，如9999）
    new_user_id = 9999
    
    # 获取所有笑话ID
    all_joke_ids = jokes_df['joke_id'].unique()
    
    # 获取用户未评分的笑话
    unrated_jokes = [j for j in all_joke_ids if j not in user_ratings.keys()]
    
    # 预测用户对未评分笑话的评分
    predictions = []
    for joke_id in unrated_jokes:
        pred = model.predict(new_user_id, joke_id)
        predictions.append((joke_id, pred.est))
    
    # 按预测评分排序，获取Top N推荐
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    
    # 获取推荐笑话的文本
    rec_jokes = []
    for joke_id, score in recommendations:
        joke_text = jokes_df[jokes_df['joke_id'] == joke_id]['joke_text'].iloc[0]
        rec_jokes.append((joke_id, joke_text, score))
    
    return rec_jokes

def display_recommendations(model, ratings_df, jokes_df):
    """显示推荐结果"""
    if st.button("获取Surprise推荐"):
        if len(st.session_state.user_ratings) >= 3:
            recommendations = generate_surprise_recommendations(
                st.session_state.user_ratings, model, ratings_df, jokes_df
            )
            
            st.header("Surprise推荐笑话")
            st.session_state.recommendations = recommendations
            
            for i, (joke_id, text, score) in enumerate(recommendations):
                st.subheader(f"推荐 {i+1}")
                st.write(text)
                st.write(f"预测评分：{score:.2f}")
                
                # 为推荐的笑话添加评分功能
                rating = st.slider(
                    f"为推荐笑话 {i+1} 评分", 
                    min_value=-10.0, 
                    max_value=10.0, 
                    value=0.0, 
                    step=0.1,
                    key=f"rec_{joke_id}"
                )
                
                if 'rec_ratings' not in st.session_state:
                    st.session_state.rec_ratings = {}
                
                st.session_state.rec_ratings[joke_id] = rating
        
        else:
            st.warning("请先为至少3个笑话评分！")

def calculate_satisfaction(rec_ratings, rating_range=(-10, 10)):
    """计算推荐满意度"""
    if not rec_ratings:
        return 0.0
    
    min_r, max_r = rating_range
    avg_rating = np.mean(list(rec_ratings.values()))
    
    # 归一化满意度计算
    satisfaction = ((avg_rating - min_r) / (max_r - min_r)) * 100
    return max(0, min(100, satisfaction))

def display_satisfaction():
    """显示满意度计算结果"""
    if st.button("计算推荐满意度"):
        if 'rec_ratings' in st.session_state and st.session_state.rec_ratings:
            satisfaction = calculate_satisfaction(st.session_state.rec_ratings)
            
            st.header("推荐满意度")
            st.write(f"平均评分：{np.mean(list(st.session_state.rec_ratings.values())):.2f}")
            st.write(f"满意度：{satisfaction:.1f}%")
            st.write("解释：基于评分范围[-10, 10]归一化，反映推荐符合您喜好的程度。")
        else:
            st.warning("请先为推荐的笑话评分！")

def main():
    st.set_page_config(page_title="Surprise笑话推荐", layout="wide")
    st.title="基于Surprise的协同过滤推荐系统"
    
    # 加载模型和数据
    model = load_trained_model()
    if not model:
        st.stop()
    
    try:
        ratings_df, jokes_df = load_experiment_data()
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        st.stop()
    
    # 侧边栏显示数据统计
    with st.sidebar:
        st.header("数据统计")
        st.write(f"评分记录数: {len(ratings_df)}")
        st.write(f"笑话数量: {len(jokes_df)}")
        st.write(f"用户数量: {ratings_df['user_id'].nunique()}")
    
    # 主内容区
    st.header("1. 为随机笑话评分")
    display_random_jokes(jokes_df)
    
    st.header("2. 获取推荐")
    display_recommendations(model, ratings_df, jokes_df)
    
    st.header("3. 评价推荐效果")
    display_satisfaction()

if __name__ == "__main__":
    main()