"""Name, city and language pools for fictitious synthetic patients (no real individuals)."""

FIRST_NAMES = {
    "south": {
        "M": ["Arjun", "Karthik", "Suresh", "Ramesh", "Venkatesh", "Prakash", "Manjunath", "Srinivas", "Anand", "Ravi",
              "Mahesh", "Ganesh", "Naveen", "Harish", "Murali", "Balaji", "Rajesh", "Vijay", "Sanjay", "Deepak"],
        "F": ["Lakshmi", "Priya", "Kavya", "Divya", "Meena", "Anitha", "Shobha", "Revathi", "Deepa", "Sujatha",
              "Padma", "Geetha", "Radha", "Sangeetha", "Vidya", "Bhavana", "Pooja", "Asha", "Nirmala", "Usha"],
    },
    "north": {
        "M": ["Rahul", "Amit", "Vikas", "Rohit", "Sandeep", "Manoj", "Rajiv", "Ashok", "Pankaj", "Gaurav",
              "Harpreet", "Gurpreet", "Anil", "Sunil", "Yogesh", "Alok", "Nitin", "Vivek", "Ajay", "Mohit"],
        "F": ["Neha", "Pooja", "Sunita", "Anjali", "Kiran", "Rekha", "Seema", "Priyanka", "Ritu", "Swati",
              "Simran", "Manpreet", "Meenakshi", "Asha", "Nisha", "Shalini", "Komal", "Rashmi", "Vandana", "Poonam"],
    },
    "west": {
        "M": ["Sachin", "Rahul", "Nilesh", "Mahesh", "Ketan", "Jignesh", "Hardik", "Prashant", "Sagar", "Amol",
              "Tushar", "Parag", "Chetan", "Mayur", "Kunal", "Vishal", "Hemant", "Dinesh", "Rakesh", "Umesh"],
        "F": ["Snehal", "Pooja", "Madhuri", "Ashwini", "Hetal", "Komal", "Rupali", "Sneha", "Pallavi", "Jyoti",
              "Manisha", "Varsha", "Shital", "Dipali", "Kavita", "Nayana", "Bhavna", "Rina", "Archana", "Prachi"],
    },
    "east": {
        "M": ["Sourav", "Arindam", "Subhash", "Debashis", "Abhijit", "Sanjib", "Partha", "Biswajit", "Tapan", "Amitabh",
              "Pranab", "Rajat", "Sudip", "Manas", "Bikash", "Ranjan", "Anirban", "Sumanta", "Tarun", "Dipankar"],
        "F": ["Rupa", "Moumita", "Sharmila", "Ananya", "Puja", "Rituparna", "Sutapa", "Madhumita", "Payel", "Soma",
              "Mitali", "Rina", "Tanushree", "Swagata", "Debjani", "Ipsita", "Sudeshna", "Mousumi", "Anwesha", "Jayeeta"],
    },
}

SURNAMES = {
    "south": ["Rao", "Reddy", "Iyer", "Nair", "Gowda", "Shetty", "Pillai", "Menon", "Naidu", "Hegde",
              "Krishnan", "Subramanian", "Kumar", "Murthy", "Bhat", "Acharya", "Raju", "Prasad", "Varma", "Shenoy"],
    "north": ["Sharma", "Verma", "Gupta", "Singh", "Yadav", "Agarwal", "Mishra", "Pandey", "Chauhan", "Kapoor",
              "Malhotra", "Saxena", "Tiwari", "Srivastava", "Bansal", "Sethi", "Arora", "Bhatia", "Dubey", "Rawat"],
    "west": ["Patil", "Deshmukh", "Joshi", "Kulkarni", "Shah", "Patel", "Mehta", "Desai", "Pawar", "Jadhav",
             "Shinde", "Gaikwad", "Trivedi", "Parikh", "Bhosale", "More", "Chavan", "Thakkar", "Kale", "Naik"],
    "east": ["Chatterjee", "Banerjee", "Mukherjee", "Das", "Bose", "Ghosh", "Sen", "Dutta", "Roy", "Mohanty",
             "Mishra", "Sahoo", "Pattnaik", "Barua", "Bora", "Saha", "Mondal", "Chakraborty", "Pal", "Nayak"],
}

CITIES = {
    "south": ["Bengaluru", "Chennai", "Hyderabad", "Kochi", "Mysuru", "Coimbatore", "Visakhapatnam", "Mangaluru"],
    "north": ["Delhi", "Lucknow", "Jaipur", "Chandigarh", "Kanpur", "Ludhiana", "Varanasi", "Dehradun"],
    "west": ["Mumbai", "Pune", "Ahmedabad", "Surat", "Nagpur", "Vadodara", "Nashik", "Goa"],
    "east": ["Kolkata", "Bhubaneswar", "Guwahati", "Patna", "Ranchi", "Siliguri", "Cuttack", "Durgapur"],
}

LANGUAGES = {
    "south": ["Kannada", "Tamil", "Telugu", "Malayalam"],
    "north": ["Hindi", "Hindi", "Punjabi"],
    "west": ["Marathi", "Gujarati", "Hindi"],
    "east": ["Bengali", "Odia", "Assamese"],
}
