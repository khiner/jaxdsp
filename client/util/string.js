// E.g. long_parameter_name => Long parameter name
export const snakeCaseToSentence = name =>
  name
    ?.split('_')
    .join(' ')
    .replace(/^(.)/, firstLetter => firstLetter.toUpperCase())
