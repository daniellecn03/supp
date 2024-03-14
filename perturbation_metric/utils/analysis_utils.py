import os 
import pickle
from perturbation_metric.attributions import vit_attributions, resnet_attributions
from perturbation_metric.perturbation import classic_perturbation, inpaint_perturbation
from perturbation_metric.utils import model

def get_gap(val):
	if val < 0:
		return 1
	else:
		return 0
	
def get_series_del(d, factor) :
	result = []
	for step in ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']:
		result.append(d[step]/factor)
	return result

def get_original_prediction_resnet(model, image_path):
	normalize = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
	)])

	image = normalize(Image.open(image_path))
	image = torch.unsqueeze(image, dim=0).cuda()

	model_output = model.forward(image).cpu().detach()
	conf = softmax(model_output)

	top1 = torch.topk(model_output[0],2).indices[0].item()
	top2 = torch.topk(model_output[0],2).indices[1].item()
	top10 = torch.topk(model_output[0],10).indices[9].item()
	top20 = torch.topk(model_output[0],20).indices[10].item()
	
	return top1, get_conf(
		conf[0][top1]), top2, get_conf(conf[0][top2]), top10, get_conf(conf[0][top10])

# OUR APPROACH
def get_inpaint_stats(filenames, directory, attribution_methods, top2_corr):
    dict_={}
    for attr in attribution_methods:
        dict_[attr] = {}
        for filename in filenames:
            dict_[attr][filename] = {}
            for step in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']:
                dict_[attr][filename][step] = 0

    for filename in filenames:
        path = directory + filename
        top1 = int(filename.split('_')[0])
        with open(path + "/new_results.pkl", 'rb') as f:
            results = pickle.load(f)
        for r in results:
            attr = r[1]
            if attr not in attribution_methods:
                continue
            step = str(r[2])
            idx = r[3]
            if(bool(r[9])):
                key = '%s_%s_%s_%s'% (filename, attr, step, idx)
                dict_[attr][filename][step] += (((top2_corr[key] / (float(step)*224*224))))
                
    def get_series(d) :
        result = []
        for step in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']:
            val = d[step]/3
            result.append(val)    
        return result

    dicti = {}
    for attr in attribution_methods:
        dicti[attr] = {}
        for filename in filenames:
            dicti[attr][filename] = get_series(dict_[attr][filename])

    aggregated = {}
    for attr in attribution_methods:
        values = []
        values_not_top1 = []
        for filename in filenames:
            values.append(dicti[attr][filename]) 
        aggregated[attr] = [1 - round(sum(x)/len(filenames),2) for x in zip(*values)]
        aggregated[attr].insert(0,1.0)
    return aggregated

# VIOLATION BASELINE 
def get_violations_stats(filenames, directory, attribution_methods, results_pickle, factor):   
    dict_1={}
    for attr in attribution_methods:
        dict_1[attr] = {}
        for filename in filenames:
            dict_1[attr][filename] = {}
            for step in ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']:
                dict_1[attr][filename][str(step)] = 0     
                
    for filename in filenames:
        path = directory + filename
        with open(path + "/%s.pkl" % results_pickle, 'rb') as f:
            results = pickle.load(f)
        with open(path + "/original.pkl", 'rb') as o:
            original = pickle.load(o)
        for r in results: 
            attr = r[1]
            if attr not in attribution_methods:
                continue
            step = str(r[2])
            if (results_pickle.startswith('deletion') or results_pickle.startswith('new_deletion')):
                if(directory == dogs_directory):
                    gap1 =  original[1] - r[3][5]
                else:
                    gap1 =  original[1] - r[3][9]
            dict_1[attr][filename][str(float(step))] += get_gap(gap1)
    dict_1_ = {}
    for attr in attribution_methods:
        dict_1_[attr] = {}
        for filename in filenames:
            dict_1_[attr][filename] = get_series_del(dict_1[attr][filename], factor)
            
    aggregated_1 = {}
    for attr in attribution_methods:
        values_1 = []
        for filename in filenames:
            values_1.append(dict_1_[attr][filename]) 
        aggregated_1[attr] = [round(sum(x)/len(filenames),2) for x in zip(*values_1)]
    return aggregated_1

# PERTURBATIONS BASELINE (INCLUDING POSITIVE/NEGATIVE DELETION, BLUR, MEAN)
def get_stats(top_directory, instances, attribution_methods, stats_file):
	"""
	Aggregates the classic perturbaction results into a single dictionary conatning the 
	results for each attribution method and step.
	"""
	manipulation_methods_ = ['dlr', 'dmr','blr','bmr','mlr','mmr']
	steps_full = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
	# Init stats containers
	acc1_stats = {'dlr':{},'dmr':{},'blr':{},'bmr':{}, 'mlr':{}, 'mmr':{}}
	for attr_id, method in enumerate(attribution_methods):
		for m_idx, m in enumerate(manipulation_methods_):
			acc1_stats[manipulation_methods_[m_idx]][attribution_methods[attr_id]] = [0] * len(steps_full)
	i = 0
	for filename in instances:       
		path = top_directory + filename         
		i+=1
		with open(stats_file, 'rb') as acc1_handle:
			tmp_acc1_stats = pickle.load(acc1_handle)
		for attr_id, method in enumerate(attribution_methods):
			for m_idx, m in enumerate(manipulation_methods_):
				for s_idx in range(len(steps_full)):
					value = tmp_acc1_stats[
						manipulation_methods_[m_idx]][attribution_methods[attr_id]][s_idx]
					acc1_stats[manipulation_methods_[m_idx]][attribution_methods[attr_id]][s_idx] += value

	for attr_id, method in enumerate(attribution_methods):
		for m_idx, m in enumerate(manipulation_methods_):
			for s_idx in range(len(steps_full)):
				acc1_stats[manipulation_methods_[m_idx]][attribution_methods[attr_id]][s_idx] /= i*100
	return acc1_stats

# SALIENCY BASELINE
def get_saliency_stats(top_directory, instances, attribution_methods, stats_file):
    sal = {}
    # Init stats containers
    for attr_id, method in enumerate(attribution_methods):
            sal[attribution_methods[attr_id]] = [0] * len(steps_)
    i = 0
    for filename in instances:       
        path = top_directory + filename 
        i+=1
        with open(stats_file, 'rb') as acc1_handle:
            sal_stats = pickle.load(acc1_handle)
        for attr_id, method in enumerate(attribution_methods):
            for s_idx in range(len(steps_)):
                single_stat = sal_stats[method][s_idx]
                rec_size = single_stat[0][0]* single_stat[0][1]
                a_value = rec_size / (224 * 224)
                final_sal = np.log(max(a_value, 0.05)) - np.log(single_stat[1])
                sal[attribution_methods[attr_id]][s_idx] += final_sal

    for attr_id, method in enumerate(attribution_methods):
        for s_idx in range(len(steps_)):
            sal[attribution_methods[attr_id]][s_idx] /= i*100
    return sal