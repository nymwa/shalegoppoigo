curl -s https://raw.githubusercontent.com/Ziphil/ShaleianDictionary/master/5.5.xdc | grep -oP "(?<=^\* ).*$" | head -n -5 > dict.txt
