import os
import pandas as pd

dataset_path = "D:\\University\\SEM-V\\Deep Learning\\DL Project\\Clutter_Image_Classification_Model_for_Smart_Homes\\dataset\\raw\\indoorCVPR_09\\Images"

high_clutter = {
    "artstudio","bakery","bar","bookstore","bowling","buffet","children_room",
    "clothingstore","computerroom","deli","dentaloffice","dining_room",
    "fastfood_restaurant","florist","gameroom","garage","greenhouse","grocerystore",
    "gym","hairsalon","hospitalroom","jewelleryshop","kitchen","laboratorywet",
    "laundromat","library","livingroom","mall","meeting_room","movietheater",
    "office","pantry","playroom","restaurant","restaurant_kitchen","shoeshop",
    "toystore","trainstation","videostore","warehouse","winecellar"
}

medium_clutter = {
    "auditorium","casino","classroom","cloister","closet","corridor","elevator",
    "inside_bus","inside_subway","kindergarden","lecture_room","lobby","locker_room",
    "museum","nursery","operating_room","poolinside","prisoncell","studiomusic",
    "subway","waitingroom"
}

low_clutter = {
    "airport_inside","bathroom","church_inside","concert_hall","parking_garage",
    "staircase"
}

rows = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if not os.path.isdir(category_path):
        continue
    if category in high_clutter:
        label = "high"
    elif category in medium_clutter:
        label = "medium"
    else:
        label = "low"
    for img in os.listdir(category_path):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(category_path, img)
            rows.append([img_path, label])

df = pd.DataFrame(rows, columns=["image_path", "clutter_level"])
df.to_csv("mit_indoor_clutter_dataset.csv", index=False)
print("CSV generated successfully!")
