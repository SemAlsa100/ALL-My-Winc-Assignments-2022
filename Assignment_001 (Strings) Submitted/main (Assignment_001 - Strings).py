# Do not modify these lines
__winc_id__ = '71dd124b4a6e4d268f5973db521394ee'
__human_name__ = 'strings'

# Add your code after this line

#####################################
# Part 1 of Assignment_001 (Strings)
#####################################
scorer_0 = "Ruud Gullit"
scorer_1 = "Marco van Basten"

goal_0 = 32
goal_1 = 54

scorer_string_0 = scorer_0 + " " + str(goal_0)
scorer_string_1 = scorer_1 + " " + str(goal_1)
print(scorer_string_0)
print(scorer_string_1)

scorers = scorer_string_0 + ', ' + scorer_string_1
# scorers = [scorer_string_0, scorer_string_1]
print(scorers)

# for i in range(2):
# print(f"{scorer_0} scored in the {goal_0}nd minute")
# print(f"{scorer_1} scored in the {goal_0}th minute")

report = (f"{scorer_0} scored in the {goal_0}nd minute") + \
    "\n" + (f"{scorer_1} scored in the {goal_1}th minute")
print(report)

#####################################
# Part 2 of Assignment_001 (Strings)
#####################################
# player = "Ruud Gullit"
# player = "Marco van Basten"
player = "Hans van Breukelen"

first_name = player[:player.find(" ")]
# first_name = player[:3]
print(first_name)
last_name = player[player.find(" ")+1:]
print(last_name)
# last_name_len = len(last_name)
last_name_len = len(player[player.find(" ")+1:])
# print(str(last_name_len))
print(last_name_len)
name_short = first_name[0] + '. ' + player[len(first_name)+1:]
print(name_short)

# chant = first_name + "!"
# print(chant)

chant = first_name + "!"
for i in range(len(first_name)-1):
    chant = chant + f" {first_name}!"

chant = chant.rstrip()
# chant = rstrip(chant)
print(chant)

# good_chant = chant[len(chant)-1:] != " "
good_chant = chant[-1] != " "
# good_chant = chant[-1]
print("good_chant = " + str(good_chant))
