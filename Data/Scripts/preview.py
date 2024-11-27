import mwparserfromhell
import os
import json
import pandas as pd
import bz2
from tqdm.auto import tqdm
from lxml import etree as ET

NS = '{http://www.mediawiki.org/xml/export-0.11/}' # Namespace for the XML dump

def get_top_parent(title):
    current = title
    visited = set()
    # dfs to find the top parent
    while current in parent_map and current not in visited:
        visited.add(current)
        current = parent_map[current]
    return current

if __name__ == '__main__':
    print("Starting the script...")

    dump_file = "enwikivoyage-latest-pages-articles.xml"
    dump_file = os.path.join("..", "Storage", dump_file)

    if not os.path.exists(dump_file):
        if not os.path.exists(dump_file + ".bz2"):
            raise FileNotFoundError("Dump file not found")
        else:
            # Extract the dump file
            print("Extracting dump file...")
            with open(dump_file, 'wb') as new_file, bz2.BZ2File(dump_file + ".bz2", 'rb') as file:
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(data)
            print("Dump file extracted")

    context = ET.iterparse(dump_file, events=('end',), tag=NS + 'page')

    # First pass: Build the parent map
    destinations = {}
    parent_map = {}

    for event, elem in tqdm(context, desc="Parsing XML", unit="pages"):
        ns = elem.find(NS + 'ns').text
        if ns != '0':
            elem.clear()
            continue

        title = elem.find(NS + 'title').text

        revision = elem.find(NS + 'revision')
        if revision is not None:
            text_elem = revision.find(NS + 'text')
            if text_elem is not None and text_elem.text is not None:
                text = text_elem.text

                if '{{IsPartOf' in text or '{{Pagebanner' in text:
                    wikicode = mwparserfromhell.parse(text)

                    # Check for disambiguation
                    if any(t.name.matches('disambiguation') for t in wikicode.filter_templates()):
                        continue

                    # Map parents
                    for template in wikicode.filter_templates():
                        if template.name.matches('IsPartOf'):
                            parent = template.get(1).value.strip_code().strip()
                            parent_map[title] = parent
                            break

                    # Store the wikicode for later processing
                    destinations[title] = wikicode

        elem.clear()

    # Second pass: Extract information
    destination_data = []

    for title, wikicode in tqdm(destinations.items(), desc="Extracting data", unit="pages"):
        destination_info = {
            'Name': title,
            'Continent': get_top_parent(title),
            'Introduction': None,
            'Understand': None,
            'Get in': None,
            'See': None,
            'Do': None,
            'Buy': None,
            'Eat': None,
            'Drink': None,
            'Sleep': None,
            'Stay safe': None,
            'Go next': None
        }

        # Extract sections
        for section in wikicode.get_sections(include_lead=True, flat=True):
            header = section.filter_headings()
            if header:
                heading = header[0].title.strip().lower()
                content = section.strip_code().strip()
                if heading == 'understand':
                    destination_info['Understand'] = content
                elif heading == 'get in':
                    destination_info['Get in'] = content
                elif heading == 'see':
                    destination_info['See'] = content
                elif heading == 'do':
                    destination_info['Do'] = content
                elif heading == 'buy':
                    destination_info['Buy'] = content
                elif heading == 'eat':
                    destination_info['Eat'] = content
                elif heading == 'drink':
                    destination_info['Drink'] = content
                elif heading == 'sleep':
                    destination_info['Sleep'] = content
                elif heading == 'stay safe':
                    destination_info['Stay safe'] = content
                elif heading == 'go next':
                    destination_info['Go next'] = content
                # Add more sections as needed
            else:
                # Introduction section
                intro_text = section.strip_code().strip()
                destination_info['Introduction'] = intro_text

        destination_data.append(destination_info)

    print("Data extracted")
    print("Ended up with {} destinations".format(len(destination_data)))
    print("Saving data...")

    with open(os.path.join("..", "Storage", "destination_data.json"), 'w') as f:
        json.dump(destination_data, f)

    print("Data saved")
    print("Done!")