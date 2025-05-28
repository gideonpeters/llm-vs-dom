import scipy.stats as stats
import pandas as pd
import json

results = [
    {
        "website": "youtube",
        "no_of_chunks": 3,
        "ir_after": ""
    },
    {
        "website": "facebook",
        "no_of_chunks": 2,
        "ir_after": ""
    },
    {
        "website": "twitter",
        "no_of_chunks": 2,
        "ir_after": ""
    },
    {
        "website": "linkedin",
        "no_of_chunks": 4,
        "ir_after": ""
    },
    {
        "website": "reddit",
        "no_of_chunks": 9,
        "ir_after": ""
    },
    {
        "website": "github",
        "no_of_chunks": 9,
        "ir_after": ""
    },
    {
        "website": "aliexpress",
        "no_of_chunks": 7,
        "ir_after": ""
    },
    {
        "website": "pinterest",
        "no_of_chunks": 2,
        "ir_after": ""
    },
    {
        "website": "ebay",
        "no_of_chunks": 17,
        "ir_after": ""
    },
    {
        "website": "netflix",
        "no_of_chunks": 2,
        "ir_after": ""
    },
    {
        "website": "quora",
        "no_of_chunks": 4,
        "ir_after": ""
    },
    {
        "website": "twitch",
        "no_of_chunks": 2,
        "ir_after": ""
    },
    {
        "website": "medium",
        "no_of_chunks": 3,
        "ir_after": ""
    },
    {
        "website": "walmart",
        "no_of_chunks": 10,
        "ir_after": ""
    },
    {
        "website": "airbnb",
        "no_of_chunks": 10,
        "ir_after": ""
    }
]

# path_to_results = "./../dataset/lh-modified-reports/gpt-4o-mini"
path_to_results = "./../dataset/lh-modified-reports/claude-3-7-sonnet-20250219"
path_to_original = "./../dataset/lh-original-reports"
path_to_audit_groupings = "./../results/audit_groupings.csv"

audit_groupings_df = pd.read_csv(path_to_audit_groupings)

def is_valid_audit(audit):
  if((audit['scoreDisplayMode'] == 'notApplicable') or
    (audit['scoreDisplayMode'] == 'binary' and audit['score'] == 1) or
    (audit['scoreDisplayMode'] == 'informative') or
    (audit['scoreDisplayMode'] == 'manual') or
    (audit['scoreDisplayMode'] == 'error') or
    (audit['scoreDisplayMode'] == 'metricSavings' and audit['score'] == 1) or
    (audit['scoreDisplayMode'] == 'numeric' and audit['score'] == 1)):
    return False

  return True

audit_groupings_dict = audit_groupings_df.set_index('audit_name').T.to_dict('records')[0]

all_unique_audit_categories = set(audit_groupings_df['category'].tolist())
# combine SEO and accessibility
all_unique_audit_categories = [category if category != "SEO" else "accessibility" for category in all_unique_audit_categories]
all_unique_audit_categories = set(all_unique_audit_categories)
print("There are ", len(all_unique_audit_categories), "unique audit categories in the dataset.")

for category in all_unique_audit_categories:
    all_results = pd.DataFrame(results.copy())

    for index, row in all_results.iterrows():
        if category == "SEO":
           continue

        with open(f"{path_to_results}/{row['website']}.json", 'r') as file:
            audits = json.load(file)['audits']

        with open(f"{path_to_original}/{row['website']}.json", 'r') as file:
            original_audits = json.load(file)['audits']

        audits = [audit for key, audit in audits.items() if is_valid_audit(audit)]
        original_audits = [audit for key, audit in original_audits.items() if is_valid_audit(audit)]

        audits_df = pd.DataFrame(audits)
        original_audits_df = pd.DataFrame(original_audits)

        audits_df['category'] = audits_df['id'].map(audit_groupings_dict)
        original_audits_df['category'] = original_audits_df['id'].map(audit_groupings_dict)

        compare_category = [category]
        if category == "accessibility":
            compare_category = ["accessibility", "SEO"]

        filtered_audits = audits_df[audits_df['category'].isin(compare_category)]
        filtered_original_audits = original_audits_df[original_audits_df['category'].isin(compare_category)]
    

        count_of_audits = len(filtered_audits)
        count_of_original_audits = len(filtered_original_audits)
        all_results.at[index, 'ir_after'] = count_of_audits
        all_results.at[index, 'ir_before'] = count_of_original_audits
        all_results.at[index, 'ir_diff'] = count_of_audits - count_of_original_audits

    X = all_results['no_of_chunks']
    Y = all_results['ir_diff']

    spearmanr = stats.spearmanr(X, Y, alternative='two-sided')
    print(f"For {category} category:")
    print(f"Spearman statistic: {spearmanr.statistic}, p-value: {spearmanr.pvalue}\n")   


