import re, glob
for path in glob.glob("latex/**/*.tex", recursive=True):
    with open(path, "r") as f:
        text = f.read()
    
    if "node2vec" in text.lower():
        # General blind lowercase regexes for node2vec are dangerous in LaTeX. Let's do selective replacements.
        text = re.sub(r"και οι ενσωματώσεις \\en\{Node2Vec\} που\nυπολογίζονται πάνω του", "που\nυπολογίζονται πάνω του", text)
        text = re.sub(r"Το\nπρόβλημα είναι εντονότερο για το \\en\{Node2Vec\}, αφού σχεδόν κάθε κόμβος\nεπικύρωσης έχει ήδη ενσωμάτωση\. ", "", text)
        text = re.sub(r"\\en\{Node2Vec\}\\\\(όλα", r"(όλα", text)
        text = re.sub(r"και οι ενσωματώσεις \\en\{Node2Vec\}\n\s*υπολογίζονται", "υπολογίζονται", text)
        text = re.sub(r"και οι ενσωματώσεις \\en\{Node2Vec\} εκπαιδεύονται.*?\n.*?ανέφικτη σε αυτή την κλίμακα γραφήματος\.", "", text, flags=re.DOTALL)
        
        # References/Macros
        text = re.sub(r"\\cite\{grover2016node2vec\},? ?", "", text)
        text = re.sub(r"το \\en\{Node2Vec\}\n?,? ?", "", text)
        text = re.sub(r"Οι μέθοδοι ενσωμάτωσης γραφημάτων όπως το\n?\\en\{Node2Vec\} μαθαίνουν χαμηλοδιάστατες αναπαραστάσεις που\n?καταγράφουν την τοπολογία της γειτονιάς, ωστόσο εξακολουθούν να αγνοούν το κείμενο\n?των κόμβων και απαιτούν επανεκπαίδευση κάθε φορά που αλλάζει το γράφημα\.", "", text)
        
        with open(path, "w") as f:
            f.write(text)
