from bs4 import BeautifulSoup
import requests
import time
import os
import csv

base_url = "https://icons.pstgu.ru/icon"
all_items_data = []

start_page = 1
end_page = 146 #146

def get_fragments(soup):
    # Извлекает названия фрагментов из страницы иконы
    fragments_header = soup.find('h2', string='Фрагменты')
    fragments = []
    
    if fragments_header:
        fragments_div = fragments_header.find_next_sibling('div', class_='item-list')
        if fragments_div:
            for item in fragments_div.find_all('div', class_='item'):
                title_div = item.find('div', class_='title')
                if title_div:
                    fragment_name = title_div.get_text(strip=True)
                    fragments.append(fragment_name)
    
    return fragments

with open('logs.txt', mode='w', encoding='utf-8') as log_file:
    try:
        for page_num in range(start_page, end_page + 1):
            list_url = f"{base_url}?page={page_num}&per-page=40"
            
            print(f"Парсинг страницы {page_num}: {list_url}", file=log_file)
            
            try:
                list_page = requests.get(list_url)
                list_page.raise_for_status()
                soup = BeautifulSoup(list_page.text, "html.parser")
                
                items = soup.select('#w0 div[data-key]')
                
                print(f"Найдено икон: {len(items)}", file=log_file)
                
                for item in items:
                    data_key = item['data-key']
                    item_url = f"{base_url}/{data_key}"
                    print(f"  Обрабатываем икону {data_key}: {item_url}", file=log_file)
                    
                    try:
                        item_page = requests.get(item_url)
                        item_page.raise_for_status()
                        item_soup = BeautifulSoup(item_page.text, "html.parser")

                        # ID, Название, Реставрация, Размер, Автор, Материал, Персоналии, Уточняющая датировка, Век, Страна, Место хранения, Иконография, категория, Тип изображения

                        info_block = item_soup.select_one('.col-xs-8')

                        item_data = {
                            'id': data_key,
                            'Название': ' '.join(info_block.find('h1').text.split()) if info_block and info_block.find('h1') else "None",
                        }

                        if info_block:
                            for p in info_block.select('p'):
                                strong_tag = p.find('strong')
                                if strong_tag:
                                    key = ' '.join(strong_tag.text.strip().rstrip(':').split())
                                    strong_tag.extract()
                                    value = ' '.join(p.text.strip().split())
                                    item_data[key] = value
                        
                        # Добавляем фрагменты к данным иконы
                        fragments = get_fragments(item_soup)
                        item_data['Фрагменты'] = ', '.join(fragments) if fragments else 'None'

                        # Скачиваем изображение
                        download_link = item_soup.find('a', attrs={'download': True})
                        if download_link and download_link.get('href'):
                            image_url = f"https://icons.pstgu.ru{download_link['href']}"
                            
                            ext = os.path.splitext(download_link['href'])[1] or '.jpg'
                            img_filename = f"images/{data_key}{ext}"
                            
                            try:
                                img_response = requests.get(image_url, stream=True, timeout=10)
                                img_response.raise_for_status()
                                
                                with open(img_filename, 'wb') as img_file:
                                    for chunk in img_response.iter_content(1024):
                                        img_file.write(chunk)
                                
                                # print(f"    Изображение сохранено: {img_filename}")
                            except Exception as e:
                                print(f"    Ошибка при скачивании изображения: {str(e)}", file=log_file)
                        else:
                            print("    Ссылка на изображение не найдена", file=log_file)
                        
                        all_items_data.append(item_data)
                        # print(f"    Собрано: {item_data['Название']}", file=log_file)
                        
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"    Ошибка при обработке иконы {data_key}: {str(e)}", file=log_file)
                        continue
                    # break
                        
            except Exception as e:
                print(f"Ошибка при загрузке страницы списка {page_num}: {str(e)}", file=log_file)
                continue
    except Exception as e:
        print(f"Ошибка при записи логов: {str(e)}", file=log_file)

csv_headers = [
    'id', 'Название', 'Реставрация', 'Размер', 'Автор', 'Материал',
    'Персоналии', 'Уточняющая датировка', 'Век', 'Страна',
    'Место хранения', 'Иконография, категория', 'Тип изображения', 'Фрагменты'
]

processed_data = []
for item in all_items_data:
    processed_item = {}
    for header in csv_headers:
        value = item.get(header, None)

        if value is not None:
            value = value.strip().replace(";", ",")
            if value.lower() in ("нет данных", "н/д", "не указано"):
                value = None
        processed_item[header] = str(value)
    processed_data.append(processed_item)


csv_filename = 'icons_dataset.csv'
with open(csv_filename, mode='w', encoding='utf-8', newline='') as file:
    csv_writer = csv.DictWriter(file, fieldnames=csv_headers, delimiter = ";", quotechar="'")
    csv_writer.writeheader()
    csv_writer.writerows(processed_data)

print(f"\nДанные сохранены в файл: {csv_filename}")
print(f"Всего записей: {len(processed_data)}")