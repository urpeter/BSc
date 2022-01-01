from ast import literal_eval


with open("corpora.txt","r") as f:
    whole_text = f.read()
    new_t = literal_eval(whole_text)
    lan_counter = 0
    counter = 0
    t = 0
    for lan in new_t:
        c = 0
        for tuple in new_t[lan]:
            lan_counter+=1
            if tuple[2] == "UNKNOWN":
                c += 1
                t += 1
            counter += 1

        print(str(lan) +": " + str(lan_counter) +" "+ str(c) )
        lan_counter = 0
    print(counter)
    print(t)