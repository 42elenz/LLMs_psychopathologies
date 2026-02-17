library(tidyverse)
library(lme4)
library(dplyr)
library(furrr)

df2 <- read_csv("disagreement_data.csv") %>% 
  select(-c("final_value", "correct_reference")) %>% 
  mutate(strategy = as.factor(strategy_chosen), item_id = as.factor(item_id), video_id = as.factor(video_id), pair_id=as.factor(pair_id)) %>% 
  select(-strategy_chosen)

##Quasi experiment synergistic strategies##

obs_stat <- df2 %>%
  group_by(strategy) %>%
  summarise(acc = mean(as.numeric(is_correct))) %>%
  pivot_wider(names_from = strategy, values_from = acc) %>%
  mutate(
    llm_vs_random   = ai_always - human_only,
    third_vs_random = human_only_supervision - human_only
  ) %>%
  select(llm_vs_random, third_vs_random)

## Permutation function
perm_test_video <- function(df, video, B = 5) {
  
  df_v <- df %>% filter(video_id == video)
  
  obs <- df_v %>%
    group_by(strategy) %>%
    summarise(acc = mean(is_correct), .groups = "drop") %>%
    pivot_wider(names_from = strategy, values_from = acc) %>%
    mutate(
      llm_vs_human   = ai_always - human_only,
      third_vs_human = human_only_supervision - human_only
    )
  
  permute_once_v <- function() {
    df_v %>%
      group_by(video_id, item_id, pair_id) %>%
      mutate(
        strategy = if (n_distinct(is_correct) > 1)
          sample(strategy)
        else
          strategy
      ) %>%
      ungroup()
  }
  
  perm <- replicate(B, {
    df_p <- permute_once_v()
    
    df_p %>%
      group_by(strategy) %>%
      summarise(acc = mean(is_correct), .groups = "drop") %>%
      pivot_wider(names_from = strategy, values_from = acc) %>%
      mutate(
        llm_vs_human   = ai_always - human_only,
        third_vs_human = human_only_supervision - human_only
      )
  }, simplify = FALSE) %>% bind_rows()
  
  tibble(
    video_id = video,
    contrast = c("LLM vs human", "Supervisor vs human"),
    estimate = c(obs$llm_vs_human, obs$third_vs_human),
    p_value = c(
      (sum(abs(perm$llm_vs_human) >= abs(obs$llm_vs_human)) + 1) / (B + 1),
      (sum(abs(perm$third_vs_human) >= abs(obs$third_vs_human)) + 1) / (B + 1)
    )
  )
}


##Running permutation test
set.seed(1)
plan(multisession)

results_video <- future_map_dfr(
  unique(df2$video_id),
  ~ perm_test_video(df2, .x, B = 5000),
  .options = furrr_options(seed = TRUE)
)

##Save results
saveRDS(results_video, file = "results_video.rds")
