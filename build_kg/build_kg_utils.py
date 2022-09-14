#encoding:utf8
import os
import re
import json
import codecs
import threading
from py2neo import Graph
import pandas as pd 
import numpy as np 
from tqdm import tqdm 

def print_data_info(data_path):
    triples = []
    i = 0
    with open(data_path,'r',encoding='utf8') as f:
        for line in f.readlines():
            data = json.loads(line)
            print(json.dumps(data, sort_keys=True, indent=4, separators=(', ', ': '),ensure_ascii=False))
            i += 1
            if i >=5:
                break
    return triples

class MedicalExtractor(object):
    def __init__(self):
        super(MedicalExtractor, self).__init__()
        self.graph = Graph( 
            "bolt://localhost:7687",
            auth=("neo4j", "kbqa")
        )

        # 共8类节点
        self.drugs = [] # 药品
        self.recipes = [] #菜谱
        self.foods = [] #　食物
        self.checks = [] # 检查
        self.departments = [] #科室
        self.producers = [] #药企
        self.diseases = [] #疾病
        self.symptoms = []#症状

        self.disease_infos = []#疾病信息

        # 构建节点实体关系
        self.rels_department = [] #　科室－科室关系
        self.rels_not_eat = [] # 疾病－忌吃食物关系
        self.rels_do_eat = [] # 疾病－宜吃食物关系
        self.rels_recommend_eat = []  # 疾病－推荐吃食物关系
        self.rels_common_drug = []  # 疾病－通用药品关系
        self.rels_recommend_drug = [] # 疾病－热门药品关系
        self.rels_check = [] # 疾病－检查关系
        self.rels_drug_producer = [] # 厂商－药物关系

        self.rels_symptom = [] #疾病症状关系
        self.rels_accompany = [] ### edited # 疾病并发关系
        self.rels_category = [] #　疾病与科室之间的关系
        
    def extract_triples(self,data_path):
        print("从json文件中转换抽取三元组")
        with open(data_path,'r',encoding='utf8') as f:
            for line in tqdm(f.readlines(),ncols=80):
                data_json = json.loads(line)

                # 疾病节点的attributes
                disease_dict = {}

                disease_dict['name'] = ''
                disease_dict['desc'] = ''
                disease_dict['prevent'] = ''
                disease_dict['cause'] = ''
                disease_dict['easy_get'] = ''
                disease_dict['cure_way'] = ''
                disease_dict['cure_lasttime'] = ''
                disease_dict['cured_prob'] = ''

                disease_dict['get_way'] = ''
                disease_dict['cost_money'] = ''
                disease_dict['get_prob'] = ''

                if 'name' in data_json: # 疾病名称
                    disease = data_json['name']
                    disease_dict['name'] = disease
                    self.diseases.append(disease)

                if 'symptom' in data_json: # 症状
                    self.symptoms += data_json['symptom']
                    for symptom in data_json['symptom']:
                        self.rels_symptom.append([disease,'has_symptom', symptom])

                if 'acompany' in data_json: # 伴随病症
                    for acompany in data_json['acompany']:
                        self.rels_accompany.append([disease,'acompany_with', acompany])
                        self.diseases.append(acompany)

                if 'desc' in data_json: # 介绍
                    disease_dict['desc'] = data_json['desc']

                if 'prevent' in data_json: # 预防措施
                    disease_dict['prevent'] = data_json['prevent']

                if 'cause' in data_json: # 病因
                    disease_dict['cause'] = data_json['cause']

                if 'get_prob' in data_json: # 患病概率
                    disease_dict['get_prob'] = data_json['get_prob']
                
                if 'get_way' in data_json: # 传染途径
                    disease_dict['get_way'] = data_json['get_way']
                
                if 'cost_money' in data_json: # 预计治疗费用
                    disease_dict['cost_money'] = data_json['cost_money']

                if 'easy_get' in data_json: # 易感人群
                    disease_dict['easy_get'] = data_json['easy_get']

                if 'cure_department' in data_json: # 问诊科室
                    cure_department = data_json['cure_department']
                    if len(cure_department) == 1:
                         self.rels_category.append([disease, 'cure_department',cure_department[0]])
                    if len(cure_department) == 2:
                        big = cure_department[0]
                        small = cure_department[1]
                        self.rels_department.append([small,'belongs_to', big])
                        self.rels_category.append([disease,'cure_department', small])

                    self.departments += cure_department

                if 'cure_way' in data_json: # 治疗方式
                    disease_dict['cure_way'] = data_json['cure_way']

                if  'cure_lasttime' in data_json: # 预计治愈时间
                    disease_dict['cure_lasttime'] = data_json['cure_lasttime']

                if 'cured_prob' in data_json: # 治愈概率
                    disease_dict['cured_prob'] = data_json['cured_prob']

                if 'common_drug' in data_json: # 常用药物
                    common_drug = data_json['common_drug']
                    for drug in common_drug:
                        self.rels_common_drug.append([disease,'has_common_drug', drug])
                    self.drugs += common_drug

                if 'recommand_drug' in data_json: # 推荐药物
                    recommand_drug = data_json['recommand_drug']
                    self.drugs += recommand_drug
                    for drug in recommand_drug:
                        self.rels_recommend_drug.append([disease,'recommand_drug', drug])

                if 'not_eat' in data_json: # 忌口
                    not_eat = data_json['not_eat']
                    for _not in not_eat:
                        self.rels_not_eat.append([disease,'not_eat', _not])

                    self.foods += not_eat

                if 'do_eat' in data_json: # 推荐食物
                    do_eat = data_json['do_eat']
                    for _do in do_eat:
                        self.rels_do_eat.append([disease,'do_eat', _do])

                    self.foods += do_eat

                if 'recommand_eat' in data_json: # 推荐菜品
                    recommand_eat = data_json['recommand_eat']
                    for _recommand in recommand_eat:
                        self.rels_recommend_eat.append([disease,'recommand_recipes', _recommand])
                    self.recipes += recommand_eat

                if 'check' in data_json: # 检查
                    check = data_json['check']
                    for _check in check:
                        self.rels_check.append([disease, 'need_check', _check])
                    self.checks += check

                if 'drug_detail' in data_json: # 药物细节（药厂+药名）
                    for det in data_json['drug_detail']:
                        det_spilt = det.split('(')
                        if len(det_spilt) == 2:
                            p,d = det_spilt
                            d = d.rstrip(')')
                            if p.find(d) > 0: # p == producer + drug name
                                p = p[:p.find(d)]
                            elif p.find(d[0:3]) > 0: # p == producer + partial drug name
                                p = p.rstrip(d)

                            self.producers.append(p)
                            self.drugs.append(d)
                            self.rels_drug_producer.append([p,'production',d])
                        else:
                            d = det_spilt[0]
                            self.drugs.append(d)

                self.disease_infos.append(disease_dict)

    def write_nodes(self,entitys,entity_type):
        print("写入 {0} 实体".format(entity_type))
        for node in tqdm(set(entitys),ncols=80):
            cql = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(
                label=entity_type,entity_name=node.replace("'",""))
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)
        
    def write_edges(self,triples,head_type,tail_type):
        print("写入 {0} 关系".format(triples[0][1]))
        for head,relation,tail in tqdm(triples,ncols=80):
            cql = """MATCH(p:{head_type}{{name:'{head}'}})
                    WITH p
                    MATCH (q:{tail_type}{{name:'{tail}'}})
                    MERGE (p)-[r:{relation}]->(q)""".format(
                        head_type=head_type,tail_type=tail_type,head=head.replace("'",""),
                        tail=tail.replace("'",""),relation=relation)
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def set_attributes(self,entity_infos,etype):
        print("写入 {0} 实体的属性".format(etype))
        for e_dict in tqdm(entity_infos,ncols=80):
            name = e_dict['name']
            del e_dict['name']
            for k,v in e_dict.items():
                if k in ['cure_way']: 
                    cql = """MATCH (n:{label}{{name:'{name}'}})
                        set n.{k}={v}""".format(label=etype,name=name.replace("'",""),k=k,v=v)
                else:
                    cql = """MATCH (n:{label}{{name:'{name}'}})
                        set n.{k}='{v}'""".format(label=etype,name=name.replace("'",""),k=k,v=v.replace("'","").replace("\n",""))
                try:
                    self.graph.run(cql)
                except Exception as e:
                    print(e)
                    print(cql)


    def create_graph_entitys(self):
        self.write_nodes(self.drugs,'药品')
        self.write_nodes(self.recipes,'菜谱')
        self.write_nodes(self.foods,'食物')
        self.write_nodes(self.checks,'检查')
        self.write_nodes(self.departments,'科室')
        self.write_nodes(self.producers,'药企')
        self.write_nodes(self.diseases,'疾病')
        self.write_nodes(self.symptoms,'症状')

    def create_graph_relations(self):
        self.write_edges(self.rels_department,'科室','科室')
        self.write_edges(self.rels_not_eat,'疾病','食物')
        self.write_edges(self.rels_do_eat,'疾病','食物')
        self.write_edges(self.rels_recommend_eat,'疾病','菜谱')
        self.write_edges(self.rels_common_drug,'疾病','药品')
        self.write_edges(self.rels_recommend_drug,'疾病','药品')
        self.write_edges(self.rels_check,'疾病','检查')
        self.write_edges(self.rels_drug_producer,'药企','药品')
        self.write_edges(self.rels_symptom,'疾病','症状')
        self.write_edges(self.rels_accompany,'疾病','疾病')
        self.write_edges(self.rels_category,'疾病','科室')

    def set_diseases_attributes(self): 
        t=threading.Thread(target=self.set_attributes,args=(self.disease_infos,"疾病"))
        t.setDaemon(False)
        t.start()


    def export_data(self,data,path):
        if isinstance(data[0],str):
            data = sorted([d.strip("...") for d in set(data)])
        elif isinstance(data[0],list):
            data = [list(j) for j in set([tuple(i) for i in data])]
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def export_entitys_relations(self):
        self.export_data(self.drugs,'./graph_data/drugs.json')
        self.export_data(self.recipes,'./graph_data/recipes.json')
        self.export_data(self.foods,'./graph_data/foods.json')
        self.export_data(self.checks,'./graph_data/checks.json')
        self.export_data(self.departments,'./graph_data/departments.json')
        self.export_data(self.producers,'./graph_data/producers.json')
        self.export_data(self.diseases,'./graph_data/diseases.json')
        self.export_data(self.symptoms,'./graph_data/symptoms.json')

        self.export_data(self.rels_department,'./graph_data/rels_department.json')
        self.export_data(self.rels_not_eat,'./graph_data/rels_not_eat.json')
        self.export_data(self.rels_do_eat,'./graph_data/rels_do_eat.json')
        self.export_data(self.rels_recommend_eat,'./graph_data/rels_recommend_eat.json')
        self.export_data(self.rels_common_drug,'./graph_data/rels_common_drug.json')
        self.export_data(self.rels_recommend_drug,'./graph_data/rels_recommend_drug.json')
        self.export_data(self.rels_check,'./graph_data/rels_check.json')
        self.export_data(self.rels_drug_producer,'./graph_data/rels_drug_producer.json')
        self.export_data(self.rels_symptom,'./graph_data/rels_symptom.json')
        self.export_data(self.rels_accompany,'./graph_data/rels_accompany.json')
        self.export_data(self.rels_category,'./graph_data/rels_category.json')





if __name__ == '__main__':
    path = "./data/medical.json"
    # print_data_info(path)
    # extractor = MedicalExtractor()
    # extractor.extract_triples(path)
    # extractor.create_graph_entitys()
    # extractor.create_graph_relations()
    # extractor.set_diseases_attributes()
    # extractor.export_entitys_relations()
