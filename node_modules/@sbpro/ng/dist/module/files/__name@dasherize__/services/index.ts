import { <%= classify(name) %>Service } from './<%= dasherize(name) %>.service';

export const services = [<%= classify(name) %>Service];

export * from './<%= dasherize(name) %>.service';
