#app.py
from PIL import Image
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import lightgbm as lgb
import re
from sklearn.model_selection import KFold
import joblib
import datetime
from matplotlib import font_manager
import joblib
import json
import logging
font_path = 'C:/Windows/Fonts/malgun.ttf'
fontprop = font_manager.FontProperties(fname=font_path)
logging.basicConfig(level=logging.INFO)  # 로그 레벨을 INFO로 설정
logger = logging.getLogger(__name__)  # 현재 모듈의 로거 생성

app = Flask(__name__)



model = joblib.load('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/model1.pkl')
df = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/encoded_df.csv',index_col=0)
df_traded = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/df_traded.csv',index_col=0)
df_apt_addr_name = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/apt_addr_name.csv',index_col=0)
df_loc = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/loc1.csv',index_col=0)
df_cen = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/행정복지센터.csv',index_col=0, encoding='cp949')
df_cen1 = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/시구청위치.csv',index_col=0, encoding='cp949')
df_apt_name_loc = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/apt_loc_name.csv',index_col=0)
label_dict = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/apt_dict.csv',index_col=0)
df_rec_trans = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/apt_recent_trans.csv',index_col=0)
df_apt_area = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/apt_area.csv',index_col=0)






@app.route('/')
def index():
    df_apt_addr_name_json = df_apt_addr_name.to_json(orient='records')
    df_loc_json = df_loc.to_json(orient='records')
    return render_template('index1.html', apartments=df_apt_addr_name_json, coordinates=df_loc_json)#apartments=df_apt_addr_name_json, coordinates=df_loc_json

@app.route('/get_areas', methods=['POST'])
def get_areas():
    apartment = request.form['apartment']  # 클라이언트로부터 선택한 아파트 이름 받기

    areas = df_apt_area.loc[df_apt_area['단지명'] == apartment, '전용면적(㎡)'].tolist()  # 선택한 아파트의 면적 리스트 가져오기

    return {'areas': areas}


@app.route('/search', methods=['POST'])
def search():
    search_term = request.form['searchTerm']

    if search_term == '':
        # 검색어가 없는 경우 빈 추천 목록을 반환하지 않고, 'No suggestions'를 반환
        return jsonify({'suggestions': ['No suggestions']})
    
    # 판다스 데이터프레임에서 검색어 추천 목록 생성
    df = pd.read_csv('C:/Users/hyung/OneDrive/바탕 화면/capstone_aptprice/data/apt_name.csv',index_col=0)
    suggestions = df[df['단지명'].str.contains(search_term)]['단지명'].tolist()
    suggestions = suggestions[:10]  # 최대 10개의 추천 목록 표시

    return jsonify({'suggestions': suggestions})


@app.route('/get_positions')
def get_positions():
    #logger.info('GET /get_positions called')  # 로그 출력
    bounds = request.args.get('bounds')  # 클라이언트에서 보낸 화면 경계 좌표를 받음
    min_lat, max_lat, min_lng, max_lng = map(float, bounds.split(','))  # 경계 좌표를 분리하여 변수에 저장
    zoom_level = int(request.args.get('zoom'))  # 클라이언트에서 보낸 지도의 줌 레벨을 받음
    
    #app.logger.debug(f'Bounds: {bounds}')
    #app.logger.debug(f'Min Latitude: {min_lat}')
    #app.logger.debug(f'Max Latitude: {max_lat}')
    #app.logger.debug(f'Min Longitude: {min_lng}')
    #app.logger.debug(f'Max Longitude: {max_lng}')
    #app.logger.debug(f'Zoom Level: {zoom_level}')
    
    # df_loc 데이터프레임에서 경계 내에 있는 좌표만 필터링하여 가져옴
    if zoom_level > 15:
        df_positions = df_apt_name_loc[(df_apt_name_loc['y'].between(min_lat, max_lat)) & (df_apt_name_loc['x'].between(min_lng, max_lng))][['y', 'x', '단지명']]
        df_positions = pd.merge(df_positions, df_rec_trans, on='단지명', how='left')
        df_positions = df_positions.to_dict(orient='records')
        
    elif 13 < zoom_level <= 15:
        df_positions = df_cen[(df_cen['y'].between(min_lat, max_lat)) & (df_cen['x'].between(min_lng, max_lng))][['y', 'x', 'si','gu','dong','dong1']].to_dict(orient='records')
        df_positions2 = pd.DataFrame(columns=['y', 'x', 'name'])
        
        for row in df_positions:
            if pd.isna(row['dong']):
                row['dong'] = ''
            if pd.isna(row['dong1']):
                row['dong1'] = ''
                
            if row['dong'].endswith('동'):
                df_positions2 = pd.concat([df_positions2, pd.DataFrame({'y': [row['y']], 'x': [row['x']], 'name': [row['dong']]})])
            elif row['dong'].endswith('구'):
                df_positions2 = pd.concat([df_positions2, pd.DataFrame({'y': [row['y']], 'x': [row['x']], 'name': [row['dong1']]})])

        df_positions = df_positions2[df_positions2['name'] != ''].to_dict(orient='records')
    
    elif zoom_level <= 13:
        df_positions = df_cen1[(df_cen1['y'].between(min_lat, max_lat)) & (df_cen1['x'].between(min_lng, max_lng))][['y', 'x', 'si','gu','dong','dong1']].to_dict(orient='records')
        df_positions2 = pd.DataFrame(columns=['y', 'x', 'name'])

        
        for row in df_positions:
            if pd.isna(row['gu']):
                row['gu'] = ''
            if pd.isna(row['dong']):
                row['dong'] = ''
            if pd.isna(row['dong1']):
                row['dong1'] = ''
                
            if row['gu'].endswith('구'):
                df_positions2 = pd.concat([df_positions2, pd.DataFrame({'y': [row['y']], 'x': [row['x']], 'name': [row['gu']]})])
            else:
                if row['dong'].endswith('구'): #용인시 기흥구, 성남시 분당구
                    df_positions2 = pd.concat([df_positions2, pd.DataFrame({'y': [row['y']], 'x': [row['x']], 'name': [row['dong']]})])
                elif row['dong'] == '': #하남시 신장동, 이천시 마장면
                    df_positions2 = pd.concat([df_positions2, pd.DataFrame({'y': [row['y']], 'x': [row['x']], 'name': [row['gu']]})])

        df_positions = df_positions2[df_positions2['name'] != ''].to_dict(orient='records')
    
    
    app.logger.debug(f'Positions Count: {len(df_positions)}')
    #for position in df_positions:
    #    app.logger.debug(position)
    
    
    return jsonify({'positions': df_positions})


@app.route('/get_coordinates', methods=['POST'])
def get_coordinates():
    apartment_name = request.form['apartmentName']

    # 데이터프레임에서 아파트 이름에 해당하는 경도와 위도 가져오기
    latitude = df_apt_name_loc[df_apt_name_loc['단지명'] == apartment_name].iloc[0][3]
    longitude = df_apt_name_loc[df_apt_name_loc['단지명'] == apartment_name].iloc[0][2]
    print(latitude, longitude)
    return jsonify({'latitude': latitude, 'longitude': longitude})
    
    
    
    
@app.route('/get_future_price', methods=['POST'])
def get_future_price():
    apartment_name = request.form['apartmentName']
    area = request.form['area']

    # 분기 계절
    import datetime
    import numpy as np
    import matplotlib.pyplot as plt

    def get_season(month):
        if month in [3, 4, 5]:
            return '봄'
        elif month in [6, 7, 8]:
            return '여름'
        elif month in [9, 10, 11]:
            return '가을'
        else:
            return '겨울'

    apt = apartment_name
    apt_label = label_dict[label_dict['apt_name'] == apt]
    label_value = apt_label['enc'].values[0]
    input_row = df[df['단지명'] == label_value]
    input_row.drop(['log_price'], axis=1, inplace=True)
    input_row = input_row.iloc[0]
    input_row['층'] = 15
    input_row['전용면적(㎡)'] = area
    input_row['전용면적(㎡)'] = pd.to_numeric(input_row['전용면적(㎡)'], errors='coerce')

    # 시작 날짜 설정
    year = 2020
    month = 1
    day = 1

    # 예측 가격 계산
    dates = []
    predicted_prices = []
    actual_prices = []
    
    for _ in range(48):  # 4년 동안의 데이터
        date = datetime.datetime(year, month, day)
        input_row['계약년'] = year
        input_row['계약월'] = month
        input_row['계약일'] = day
        input_row['분기'] = (month - 1) // 3 + 1
        season = get_season(month)

        # 계절 원핫 인코딩
        input_row.loc['계절_봄'] = 0
        input_row.loc['계절_여름'] = 0
        input_row.loc['계절_가을'] = 0
        input_row.loc['계절_겨울'] = 0

        if season == '봄':
            input_row.loc['계절_봄'] = 1
        elif season == '여름':
            input_row.loc['계절_여름'] = 1
        elif season == '가을':
            input_row.loc['계절_가을'] = 1
        elif season == '겨울':
            input_row.loc['계절_겨울'] = 1

        input_data = pd.DataFrame([input_row])
        output = model.predict(input_data)
        pred_price = np.expm1(output)

        dates.append(date)
        predicted_prices.append(pred_price)


        # 다음달로 이동
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    # 실거래 데이터 가져오기
    
    df_traded['단지명'] = df_traded['단지명'].astype(str)
    df_traded['전용면적(㎡)'] = df_traded['전용면적(㎡)'].astype(float)
    #traded_data = df_traded[(df_traded['단지명'] == '은마') & (df_traded['전용면적(㎡)'] == 84.43)]
    traded_data = df_traded[(df_traded['단지명'] == apartment_name) & (df_traded['전용면적(㎡)'] == float(area))]
    traded_dates = pd.to_datetime(traded_data['계약년'].astype(str) + '-' + traded_data['계약월'].astype(str) + '-' + traded_data['계약일'].astype(str))
    actual_prices = traded_data['거래금액(만원)']
    #print(predicted_prices)
    #print(type(apartment_name))
    #print(type(area))
    #print(df_traded.info())
    #print(traded_data)

    # 그래프 생성
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    # 시작날짜와 종료날짜를 datetime 형식으로 생성
    #start_date = datetime.datetime(2020, 1, 1)
    #end_date = datetime.datetime(2023, 12, 31)

    # x축 범위 설정
    #ax.set_xlim(start_date, end_date)
    ax.plot(dates, predicted_prices, linestyle='-', color='blue', label='Predicted Price', zorder=5)
    ax.scatter(traded_dates, actual_prices, color='red', label='Actual Price', zorder=10)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('{}({}㎡형)  2023.12월 예측가:{}억'.format(apartment_name, area, (pred_price[0]/10000).round(1)), fontproperties=fontprop, fontsize=14)
    print((pred_price/10000).round(1))
    #ax.tick_params(axis='x', labelsize=8)
    #ax.tick_params(axis='y', labelsize=8)
    #ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    #ax.legend(loc='upper left', fontsize=10)
    ax.set_facecolor('whitesmoke')
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
    ax.grid(True)

    # 그래프를 이미지로 변환
    canvas = FigureCanvas(fig)
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    buffer.seek(0)

    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # JSON 형태로 응답 데이터 구성
    response_data = {
        'graphImage': encoded_image,
    }

    return jsonify(response_data)

    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8080",debug=True) #host="0.0.0.0", port="8080",