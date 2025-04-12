create database if not exists Apdm;
use Apdm;

create table if not exists brasileirao (
	
    datadojogo VARCHAR(50),
    home_team VARCHAR(50),
    home_team_state VARCHAR(50),
    away_team VARCHAR(50),
    away_team_state VARCHAR(50),
    home_goal varchar(5),
    away_goal varchar(5),
    season int ,
    round int 
);

LOAD DATA INFILE 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads\\Brasileirao_Matches.csv'
INTO TABLE brasileirao
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(@datetime, @home_team, @home_team_state, @away_team, @away_team_state, @home_goal, @away_goal, @season, @round)
SET 
    datadojogo = @datetime,
    home_team = @home_team,
    home_team_state = @home_team_state,
    away_team = @away_team,
    away_team_state = @away_team_state,
    home_goal = @home_goal,
    away_goal = @away_goal,
    season = @season,
    round = @round;


