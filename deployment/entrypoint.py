import toolforge

conn = toolforge.connect('enwiki')  # You can also use "enwiki_p"
toolforge.set_user_agent('peacock-finder')
query = "SELECT rev_id, rev_page FROM revision WHERE rev_timestamp > <FIGURE OUT> WHERE rev_minor_edit < 1 WHERE rev_deleted < 1`LIMIT 500"
with conn.cursor() as cur:
    cur.execute(query)
    cur.commit()
    rows = cur.fetchall()
    print(rows)
conn.close()
"""
idea: query the revision table
https://www.mediawiki.org/wiki/Manual:Revision_table

ignore if tagged as minor_edit, else look at rev_text_id
look up rev_text_id as old_id in text table, look at old_text
if old_text is not in external storage (how do we check this?) segment into sentences and run

query for last 500 edits every hour? something like that ? 

"""

