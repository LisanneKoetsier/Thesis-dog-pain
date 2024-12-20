import os
import csv

def create_lighter_annots(ipfile, opfile):
    ovids = []
    rows = []
    with open(ipfile, 'r') as f:
        reader = csv.DictReader(f, delimiter=' ')
        for row in reader:
            ovid = row['original_vido_id']
            labels = row['labels']
            if labels == '' or ovid in ovids:
                continue
            ovids += [ovid]
            rows += [[ovid, labels]]
            print([ovid, labels])
    header = ['video_id', 'labels']
    with open(opfile, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    ipfile = r'C:\Thesis\dog_pain_lisanne\clip_action_prediction\MSQNet\multi-label-action-main\datasets\animal_kingdom\action_recognition\annotation\val.csv'
    opfile = r'C:\Thesis\dog_pain_lisanne\clip_action_prediction\MSQNet\multi-label-action-main\datasets\animal_kingdom\action_recognition\annotation\val_light.csv'
    create_lighter_annots(ipfile, opfile)    